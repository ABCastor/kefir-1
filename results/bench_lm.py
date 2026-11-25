"""Zero-shot lm-evaluation-harness comparison: Kefir 1 and small open baselines.

Usage: python bench_lm.py MODEL_SPEC OUT_JSON [--limit N] [--device cuda]
MODEL_SPEC: kefir:/path/model.safetensors  or  hf:repo_id[@revision]
Kefir is scored through a custom LM class: trailing context whitespace moved to the continuation
(as lm-eval does), then context and continuation encoded separately with the pinned tokenizer and
concatenated, EOD (7992) prepended as the document-start token, every continuation token scored,
right padding (causal model, so padding never reaches earlier positions), FP32. Separate encoding
matters because the SuperBPE tokenizer can merge across the boundary ("Thank you for" + " your
help" encodes as "...for your" + " help" when encoded whole).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch
import torch.nn.functional as F
from lm_eval import simple_evaluate
from lm_eval.api.model import LM

TASKS = ["lambada_openai", "hellaswag", "piqa", "arc_easy", "arc_challenge",
         "winogrande", "sciq", "openbookqa", "blimp"]


def encode_pair(encode, eod: int, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """Return (whole, continuation) token ids: [eod] + encode(context) + encode(continuation).

    Trailing whitespace of the context moves to the continuation first, as in lm-eval. The two parts
    are encoded separately, so no token spans the boundary and every continuation byte is scored.
    """
    n = len(context) - len(context.rstrip())
    if n:
        continuation = context[-n:] + continuation
        context = context[:-n]
    ctx = [eod] + encode(context)
    cont = encode(continuation)
    if not cont:
        raise ValueError("empty continuation: nothing to score")
    return ctx + cont, cont


class KefirLM(LM):
    def __init__(self, weights: str, device: str, batch_size: int = 64) -> None:
        super().__init__()
        from kefir.artifact import load_weights
        from kefir.tokenizer import EOD_TOKEN_ID, load_tokenizer
        self.model, self.config, _ = load_weights(weights, device=device)
        self.tok = load_tokenizer()
        self.eod = EOD_TOKEN_ID
        self.dev = torch.device(device)
        self.bs = batch_size
        self.max_len = self.config.max_seq_len

    def _pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        return encode_pair(self.tok.encode, self.eod, context, continuation)

    @torch.no_grad()
    def _score(self, items: list[tuple[list[int], list[int]]]) -> list[tuple[float, bool]]:
        order = sorted(range(len(items)), key=lambda i: -len(items[i][0]))
        out: list[tuple[float, bool] | None] = [None] * len(items)
        for start in range(0, len(order), self.bs):
            idx = order[start:start + self.bs]
            seqs = []
            for i in idx:
                whole, cont = items[i]
                whole = whole[-(self.max_len + 1):]
                seqs.append((whole, len(cont)))
            width = max(len(s) - 1 for s, _ in seqs)
            inp = torch.full((len(seqs), width), self.eod, dtype=torch.long)
            for r, (s, _) in enumerate(seqs):
                inp[r, :len(s) - 1] = torch.tensor(s[:-1])
            logits = self.model.forward_with_aux(inp.to(self.dev))[0].float()
            logp = F.log_softmax(logits, dim=-1)
            for r, (s, ncont) in enumerate(seqs):
                target = torch.tensor(s[-ncont:], device=self.dev)
                lp = logp[r, len(s) - 1 - ncont:len(s) - 1]
                total = lp.gather(1, target[:, None]).sum().item()
                greedy = bool((lp.argmax(-1) == target).all().item())
                out[idx[r]] = (total, greedy)
        return out  # type: ignore[return-value]

    def loglikelihood(self, requests):
        return self._score([self._pair(*r.args) for r in requests])

    def loglikelihood_rolling(self, requests):
        from lm_eval import utils
        out = []
        for r in requests:
            ids = self.tok.encode(r.args[0])
            wins = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                token_list=ids, prefix_token=self.eod, max_seq_len=self.max_len, context_len=1)))
            scored = self._score([(list(c) + list(t), list(t)) for c, t in wins])
            out.append(sum(x for x, _ in scored))
        return out

    def generate_until(self, requests):
        raise NotImplementedError("not used by the selected tasks")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--batch-size", default="64")
    a = ap.parse_args()
    kind, _, ref = a.spec.partition(":")
    t0 = time.time()
    common = dict(num_fewshot=0, limit=a.limit,
                  bootstrap_iters=0, log_samples=False, random_seed=0, numpy_random_seed=1234,
                  torch_random_seed=1234)
    if kind == "kefir":
        lm = KefirLM(ref, a.device, int(a.batch_size))
    elif kind == "hf":
        from lm_eval.models.huggingface import HFLM
        trc = ref.endswith("+trc")
        repo, _, rev = ref.removesuffix("+trc").partition("@")
        kw = dict(pretrained=repo, dtype="float32", device=a.device, batch_size=a.batch_size)
        if rev:
            kw["revision"] = rev
        if trc:
            kw["trust_remote_code"] = True
        lm = HFLM(**kw)
    else:
        raise SystemExit(f"unknown model kind {kind}")
    # One task at a time: a task that cannot load (e.g. a script-based dataset) costs only itself.
    keep = {"spec": a.spec, "limit": a.limit, "results": {}, "versions": {}, "n-samples": {},
            "errors": {}, "task_seconds": {}}
    for task in a.tasks.split(","):
        t1 = time.time()
        try:
            res = simple_evaluate(model=lm, tasks=[task], **common)
            keep["results"].update(res["results"])
            keep["versions"].update(res.get("versions") or {})
            keep["n-samples"].update(res.get("n-samples") or {})
            r = res["results"].get(task, {})
            print("TASK", task, {k: v for k, v in r.items() if k.startswith(("acc", "perplexity", "bits", "word", "byte"))}, flush=True)
        except Exception as e:  # recorded, never silent
            keep["errors"][task] = f"{type(e).__name__}: {e}"[:500]
            print("TASKERROR", task, keep["errors"][task], flush=True)
        keep["task_seconds"][task] = round(time.time() - t1, 1)
        keep["seconds"] = round(time.time() - t0, 1)
        with open(a.out, "w") as f:
            json.dump(keep, f, indent=1, sort_keys=True, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
