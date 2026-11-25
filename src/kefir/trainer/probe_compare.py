"""Offline fixed-probe comparison and optional verified checkpoint-interval deltas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch

from kefir.data.src.joint_checkpoint import verify_local_manifest
from .observation import file_hash
from .state import load_trainer_state_file

_WEIGHT = re.compile(r"^blocks\.(\d+)\.ffn\.(gate_weight|up_weight|down_weight|router\.projection\.weight)$")


def _distribution(values):
    values = values.float().reshape(-1)
    if not values.numel() or not torch.isfinite(values).all():
        raise ValueError("distribution must be nonempty and finite")
    quantiles = torch.quantile(values, torch.tensor([0., .5, .9, .95, .99, 1.]))
    return {"count": values.numel(), "mean": values.mean().item(),
            "quantiles": dict(zip(("p00", "p50", "p90", "p95", "p99", "p100"),
                                  quantiles.tolist(), strict=True))}


def _load_probe(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("version") != 2:
        raise ValueError("unsupported probe schema")
    if value.get("router_scoring") not in {"softmax", "sqrt_softplus"}:
        raise ValueError("probe must declare its router scoring")
    for field in ("probe_sha256", "checkpoint_id", "checkpoint_manifest_sha256"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"probe missing {field}")
    for field in ("step", "consumed_tokens"):
        if type(value.get(field)) is not int or value[field] < 0:
            raise ValueError(f"invalid probe {field}")
    ce, mask = value.get("per_token_ce_fp32"), value.get("target_mask")
    if (not isinstance(ce, torch.Tensor) or ce.dtype != torch.float32
            or ce.ndim != 2 or not torch.isfinite(ce).all()):
        raise ValueError("CE must be finite FP32 [sequence, position]")
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != ce.shape or not mask.any()):
        raise ValueError("target_mask must match CE and contain a target")
    if not isinstance(value.get("routing"), dict):
        raise ValueError("routing must be a layer mapping")
    for name, rows in value["routing"].items():
        if not isinstance(name, str) or not isinstance(rows, list) or len(rows) != ce.shape[0]:
            raise ValueError("routing sequence/layer identity mismatch")
        geometry = None
        for row in rows:
            logits, indices = row["logits"], row["expert_indices"]
            if (logits.dtype != torch.float32 or logits.ndim != 2
                    or logits.shape[0] != ce.shape[1] or logits.shape[1] < 2
                    or not torch.isfinite(logits).all()):
                raise ValueError("router logits must be finite FP32 [position, experts]")
            if (indices.dtype != torch.int64 or indices.ndim != 2
                    or indices.shape[0] != ce.shape[1] or not 1 <= indices.shape[1] <= logits.shape[1]
                    or (indices < 0).any() or (indices >= logits.shape[1]).any()
                    or (indices.sort(dim=-1).values.diff(dim=-1) == 0).any()):
                raise ValueError("expert indices must be distinct valid int64 [position, top_k]")
            current = (logits.shape[1], indices.shape[1])
            if geometry is not None and geometry != current:
                raise ValueError("router geometry changes within a probe")
            geometry = current
    return value


def _routing_summary(churn, js, transitions):
    count = churn.numel()
    return {"token_count": count, "assignment_churn_count": int(churn.sum()),
            "assignment_churn_rate": float(churn.float().mean()) if count else 0.,
            "churn_definition": "token changed its selected expert set",
            "transition_definition": "rows=before expert, columns=after expert; paired by top-k rank",
            "transition_counts": transitions.tolist(),
            "router_affinity_js": _distribution(js) if count else None}


def _routing(before, after, *, scoring):
    def distribution(rows):
        logits = torch.cat([row["logits"] for row in rows]).double()
        if scoring == "softmax":
            return logits.softmax(-1)
        scores = torch.nn.functional.softplus(logits).sqrt()
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)

    layers, all_churn, all_js = {}, [], []
    total = None
    for name in sorted(before):
        first, second = before[name], after[name]
        a = torch.cat([row["expert_indices"] for row in first])
        b = torch.cat([row["expert_indices"] for row in second])
        p, q = distribution(first), distribution(second)
        if a.shape != b.shape or p.shape != q.shape:
            raise ValueError("probe router geometries differ")
        midpoint = (p + q) / 2
        # xlogy handles a zero probability without biasing or renormalizing p/q.
        js = .5 * ((torch.special.xlogy(p, p) - torch.special.xlogy(p, midpoint)).sum(-1)
                   + (torch.special.xlogy(q, q) - torch.special.xlogy(q, midpoint)).sum(-1))
        experts = p.shape[1]
        transitions = torch.bincount((a * experts + b).reshape(-1), minlength=experts**2).reshape(experts, experts)
        churn = a.sort(dim=-1).values.ne(b.sort(dim=-1).values).any(dim=-1)
        layers[name] = _routing_summary(churn, js, transitions)
        all_churn.append(churn)
        all_js.append(js)
        if total is None:
            total = torch.zeros_like(transitions)
        if total.shape != transitions.shape:
            raise ValueError("probe layers have different expert counts")
        total += transitions
    return {"layers": layers, "aggregate": _routing_summary(
        torch.cat(all_churn) if all_churn else torch.empty(0),
        torch.cat(all_js) if all_js else torch.empty(0),
        total if total is not None else torch.zeros((0, 0), dtype=torch.int64))}


def _checkpoint(path, probe):
    root = Path(path)
    verify_local_manifest(root)
    metadata = json.loads((root / "checkpoint.json").read_text())
    if (metadata.get("optimizer_step_complete") is not True
            or metadata.get("checkpoint_id") != probe["checkpoint_id"]
            or file_hash(root / "checkpoint.json") != probe["checkpoint_manifest_sha256"]):
        raise ValueError("probe does not bind to this complete checkpoint")
    # These are trusted local training artifacts; joint hashes detect corruption.
    state = load_trainer_state_file(root / "trainer_state.pkl", map_location="cpu")
    if (state["global_step"] != probe["step"]
            or state["consumed_tokens"] != probe["consumed_tokens"]):
        raise ValueError("probe counters differ from the checkpoint")
    identity = tuple(metadata.get(key) for key in ("configuration_hash", "source_hash", "schema_hash"))
    if any(not isinstance(item, str) or len(item) != 64 for item in identity):
        raise ValueError("checkpoint lacks joint identity")
    return identity, state["model"]


def _weight_delta(first_path, second_path, before, after):
    first_id, first = _checkpoint(first_path, before)
    second_id, second = _checkpoint(second_path, after)
    if first_id != second_id or first.keys() != second.keys():
        raise ValueError("checkpoint joint identities or model keys differ")
    groups = {}
    for name, a in first.items():
        match = _WEIGHT.fullmatch(name)
        if not match:
            continue
        b = second[name]
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"invalid checkpoint weights: {name}")
        layer, parameter = match.groups()
        slices = [("router", a, b)] if parameter == "router.projection.weight" else [
            (f"expert_{i:02d}", a[i], b[i]) for i in range(a.shape[0])
        ]
        for label, left, right in slices:
            entry = groups.setdefault((int(layer), label), {"parameter_count": 0, "weight_sq": 0., "delta_sq": 0.})
            entry["parameter_count"] += left.numel()
            entry["weight_sq"] += left.double().square().sum().item()
            entry["delta_sq"] += (right.double() - left.double()).square().sum().item()
    rows = []
    for (layer, group), value in sorted(groups.items()):
        norm, delta = value["weight_sq"] ** .5, value["delta_sq"] ** .5
        rows.append({"layer": layer, "group": group, "parameter_count": value["parameter_count"],
                     "weight_before_l2": norm, "weight_delta_l2": delta,
                     "weight_delta_over_before_l2": delta / norm if norm else None})
    return {"scope": "checkpoint interval; not individual-step optimizer updates",
            "before_step": before["step"], "after_step": after["step"], "groups": rows}


def compare_probes(before_path, after_path, *, checkpoint_before=None, checkpoint_after=None):
    before, after = _load_probe(Path(before_path)), _load_probe(Path(after_path))
    if before["router_scoring"] != after["router_scoring"]:
        raise ValueError("probe router scoring differs")
    if before["probe_sha256"] != after["probe_sha256"]:
        raise ValueError("probe_sha256 mismatch")
    if not torch.equal(before["target_mask"], after["target_mask"]):
        raise ValueError("target_mask mismatch")
    if before["routing"].keys() != after["routing"].keys():
        raise ValueError("routing layer-name mismatch")
    if after["step"] < before["step"] or after["consumed_tokens"] < before["consumed_tokens"]:
        raise ValueError("after probe precedes before probe")
    if (checkpoint_before is None) != (checkpoint_after is None):
        raise ValueError("supply both checkpoint directories")
    delta = (after["per_token_ce_fp32"] - before["per_token_ce_fp32"])[before["target_mask"]]
    return {"schema_version": 2,
            "comparison": {"before": str(before_path), "after": str(after_path),
                           "probe_sha256": before["probe_sha256"],
                           "before_step": before["step"], "after_step": after["step"]},
            "ce_delta_after_minus_before": {"position_scope": "target_mask=true only",
                                            **_distribution(delta)},
            "routing": {"position_scope": "all routed positions, including masked positions",
                        "router_scoring": before["router_scoring"],
                        **_routing(before["routing"], after["routing"], scoring=before["router_scoring"])},
            "checkpoint_weight_delta": (_weight_delta(checkpoint_before, checkpoint_after, before, after)
                                        if checkpoint_before is not None else None)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before_probe", type=Path)
    parser.add_argument("after_probe", type=Path)
    parser.add_argument("--checkpoint-before", type=Path)
    parser.add_argument("--checkpoint-after", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare_probes(args.before_probe, args.after_probe,
                            checkpoint_before=args.checkpoint_before, checkpoint_after=args.checkpoint_after)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
