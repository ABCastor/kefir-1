"""Exercise the real process entrypoint, immutable data and joint resume offline."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from kefir.trainer.state import load_trainer_state_file
from kefir.trainer.observation import _expert_statistics, _source_identity
from kefir.trainer.probe_compare import _routing, _weight_delta
from kefir.model.model import MoESmallLLM
from tests.test_moe_model import _tiny_moe_config
from tests.trainer_fixtures import payload

ROOT = Path(__file__).resolve().parents[1]


class TestMoECLIObservation(unittest.TestCase):
    def test_stacked_expert_statistics_match_parameter_and_gradient_norms(self):
        model = MoESmallLLM(_tiny_moe_config())
        self.assertIsNone(_expert_statistics(model)["blocks.0.ffn/expert_00"]["gradient_l2_after_clipping"])
        model(torch.tensor([[1, 2, 3]])).square().mean().backward()
        stats = _expert_statistics(model)
        expert = model.blocks[0].ffn
        weights = [expert.gate_weight, expert.up_weight, expert.down_weight]
        expected_weight = torch.cat([p[0].detach().reshape(-1) for p in weights]).norm()
        expected_gradient = torch.cat([p.grad[0].reshape(-1) for p in weights]).norm()
        observed = stats["blocks.0.ffn/expert_00"]
        self.assertAlmostEqual(observed["weight_l2"], expected_weight.item(), places=5)
        self.assertAlmostEqual(observed["gradient_l2_after_clipping"], expected_gradient.item(), places=5)
        self.assertEqual(observed["gradient_tensors_present"], 3)
        self.assertEqual(len(stats), model.config.n_layers * (model.config.num_experts + 1))

    def test_probe_churn_uses_expert_sets_and_actual_router_affinities(self):
        first = torch.tensor([[0., 2., -1., 3.], [2., -1., 1., .4]])
        second = first.flip(-1)
        before = {"layer": [{"logits": first, "expert_indices": torch.tensor([[1, 3], [0, 2]])}]}
        after = {"layer": [{"logits": second, "expert_indices": torch.tensor([[3, 1], [0, 1]])}]}
        summary = _routing(before, after, scoring="sqrt_softplus")["aggregate"]
        self.assertEqual(summary["token_count"], 2)
        self.assertEqual(summary["assignment_churn_count"], 1)
        self.assertEqual(summary["assignment_churn_rate"], .5)
        self.assertEqual(sum(map(sum, summary["transition_counts"])), 4)
        p, q = [torch.log1p(x.double().exp()).sqrt() for x in (first, second)]
        p, q = p / p.sum(-1, keepdim=True), q / q.sum(-1, keepdim=True)
        midpoint = (p + q) / 2
        expected = .5 * ((p * (p / midpoint).log()).sum(-1) + (q * (q / midpoint).log()).sum(-1))
        self.assertAlmostEqual(summary["router_affinity_js"]["mean"], expected.mean().item(), places=7)

    def test_checkpoint_delta_groups_each_stacked_expert(self):
        before = {f"blocks.0.ffn.{name}_weight": torch.ones(2, 2, 3)
                  for name in ("gate", "up", "down")}
        before["blocks.0.ffn.router.projection.weight"] = torch.ones(2, 2)
        after = {name: value * 2 for name, value in before.items()}
        with patch("kefir.trainer.probe_compare._checkpoint", side_effect=[(("identity",), before), (("identity",), after)]):
            result = _weight_delta("before", "after", {"step": 1}, {"step": 2})
        groups = {row["group"]: row for row in result["groups"]}
        self.assertEqual(set(groups), {"router", "expert_00", "expert_01"})
        for name, row in groups.items():
            count = 4 if name == "router" else 18
            self.assertEqual(row["parameter_count"], count)
            self.assertAlmostEqual(row["weight_before_l2"], count ** .5)
            self.assertAlmostEqual(row["weight_delta_l2"], count ** .5)
            self.assertAlmostEqual(row["weight_delta_over_before_l2"], 1.)

    def test_source_fingerprint_covers_entire_installed_namespace(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "src/kefir"
            observer = package / "trainer/observation.py"
            observer.parent.mkdir(parents=True)
            observer.write_text("# fixture")
            with patch("kefir.trainer.observation.__file__", str(observer)):
                previous = _source_identity(None)["source_tree_sha256"]
                for name in ("training.py", "model/config.py", "data/cache.py", "tokenizer/assets.json", "tokenizer/panel.npz"):
                    path = package / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture")
                    current = _source_identity(None)["source_tree_sha256"]
                    self.assertNotEqual(current, previous, name)
                    previous = current
                (package / "ignored.pyc").write_bytes(b"bytecode")
                self.assertEqual(_source_identity(None)["source_tree_sha256"], previous)

    def test_fresh_process_probe_profile_and_exact_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            shards = []
            for split, rows in (("train", 4), ("validation", 2)):
                (data / split).mkdir()
                raw = payload([[1+i, 2+i, 3+i, 4+i] for i in range(rows)])
                name = f"{split}/000.bin"
                (data / name).write_bytes(raw)
                shards.append(dict(filename=name, split=split, byte_size=len(raw),
                                   checksum=hashlib.sha256(raw).hexdigest(),
                                   sequence_count=rows, first_block_id=0,
                                   last_block_id=rows//2-1))
            (data / "manifest.json").write_text(json.dumps(dict(
                schema_version=2, sequence_format="context_plus_one", context_length=3,
                stored_tokens_per_sequence=4, sequences_per_block=2, shards=shards)))
            environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                           "OPENBLAS_NUM_THREADS": "2", "WANDB_MODE": "disabled",
                           "HF_HUB_OFFLINE": "1"}

            def run(name, steps, resume=None, extra=()):
                entrypoint = (
                    "from unittest.mock import patch; import runpy; "
                    "from tests.test_moe_model import _tiny_moe_config; "
                    "patch('kefir.model.setup.MoEModelConfig.accepted', "
                    "return_value=_tiny_moe_config()).start(); "
                    "runpy.run_module('kefir.model', run_name='__main__')"
                )
                command = [sys.executable, "-c", entrypoint, "--dataset-dir", str(data),
                           "--checkpoint-dir", str(root/name/"checkpoints"),
                           "--experiment-dir", str(root/name/"artifacts"),
                           "--steps", str(steps), "--model-size", "accepted", "--device", "cpu",
                           "--precision", "fp32", "--microbatch-size", "1",
                           "--seed", "13", "--probe-sequences", "2", "--validation-blocks", "1",
                           "--checkpoint-at-steps", "0,1,2", "--profile-at-steps", "1", *extra]
                if resume:
                    command += ["--resume", resume]
                result = subprocess.run(command, cwd=ROOT, env=environment, text=True,
                                        capture_output=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout[-2000:] + result.stderr[-6000:])
                return result

            continuous = run("continuous", 2)
            self.assertEqual(sum("\"validation\"" in line for line in continuous.stdout.splitlines()), 3)
            run("resumed", 1)
            run("resumed", 1, "step-00000001")
            states = [load_trainer_state_file(root/name/"checkpoints/step-00000002/trainer_state.pkl")
                      for name in ("continuous", "resumed")]
            self.assertEqual(states[0]["global_step"], 2)
            self.assertEqual(states[0]["consumed_tokens"], 12)

            def same(first, second):
                if isinstance(first, torch.Tensor):
                    self.assertTrue(torch.equal(first, second))
                elif isinstance(first, dict):
                    self.assertEqual(first.keys(), second.keys())
                    for key in first:
                        same(first[key], second[key])
                elif isinstance(first, (tuple, list)):
                    self.assertEqual(len(first), len(second))
                    for a, b in zip(first, second, strict=True):
                        same(a, b)
                else:
                    self.assertEqual(first, second)
            for key in ("model", "optimizer", "scheduler", "scaler", "consumed_tokens", "torch_rng_state"):
                same(states[0][key], states[1][key])
            artifacts = root/"continuous/artifacts"
            probe = torch.load(artifacts/"probe.pt", weights_only=True)
            self.assertEqual(probe["target_mask"].sum().item(), 6)
            first = torch.load(artifacts/"step-00000000-probe.pt", weights_only=True)
            last = torch.load(artifacts/"step-00000002-probe.pt", weights_only=True)
            self.assertEqual(first["probe_sha256"], last["probe_sha256"])
            self.assertEqual(last["per_token_ce_fp32"].shape, (2, 3))
            self.assertNotIn("lm_logits_fp16", last)
            self.assertEqual(len(last["routing"]), 4)
            trace = next(artifacts.glob("from-*/profile-step-00000001.json")).read_text()
            for label in ("session_step", "forward_ce", "backward", "optimizer", "module/blocks.0.ffn.router"):
                self.assertIn(label, trace)
            identity = json.loads((artifacts/"identity.json").read_text())
            self.assertEqual(identity["model"]["load_balancing"], "quantile")
            self.assertEqual(len(identity["source_tree_sha256"]), 64)
            events = [json.loads(line) for path in artifacts.glob("from-*/events.jsonl")
                      for line in path.read_text().splitlines()]
            sample = next(event for event in events if event["event"] == "training")
            self.assertTrue(sample["expert_statistics"])
            self.assertTrue(all("gradient_l2_after_clipping" in item
                                for item in sample["expert_statistics"].values()))
            compared = subprocess.run([
                sys.executable, "-m", "kefir.trainer.probe_compare",
                str(artifacts/"step-00000000-probe.pt"),
                str(artifacts/"step-00000002-probe.pt"),
                "--checkpoint-before", str(root/"continuous/checkpoints/step-00000000"),
                "--checkpoint-after", str(root/"continuous/checkpoints/step-00000002"),
            ], cwd=ROOT, env=environment, text=True, capture_output=True, timeout=120)
            self.assertEqual(compared.returncode, 0, compared.stderr)
            comparison = json.loads(compared.stdout)
            self.assertEqual(len(comparison["checkpoint_weight_delta"]["groups"]), 4 * 9)
            self.assertEqual(comparison["ce_delta_after_minus_before"]["count"], 6)
            # Observation and eval must not advance the serialized controller.
            for key, value in states[0]["model"].items():
                if key.endswith("token_exposure"):
                    self.assertEqual(value.sum().item(), 24)

    def test_dataset_verifier_import_does_not_require_torch(self):
        code = '''import sys
class DenyTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise RuntimeError('dataset verification imported torch')
sys.meta_path.insert(0, DenyTorch())
import kefir.data.src.verify
import kefir.data.incremental_cache
import kefir.data.incremental_frontier
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
