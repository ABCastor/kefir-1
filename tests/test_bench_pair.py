"""The lm-eval adapter splits context and continuation at the boundary, without torch.

`results/bench_lm.py` imports torch and lm_eval at module level, so only its `encode_pair`
function is compiled here (read with ast), with the bundled tokenizer.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from kefir.tokenizer import EOD_TOKEN_ID, load_tokenizer

BENCH = Path(__file__).resolve().parents[1] / "results" / "bench_lm.py"


def _encode_pair():
    tree = ast.parse(BENCH.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "encode_pair")
    namespace: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(BENCH), "exec"), namespace)
    return namespace["encode_pair"]


def test_boundary_merge_is_split_and_every_continuation_token_scored():
    tok = load_tokenizer()
    encode_pair = _encode_pair()
    context, continuation = "Thank you for", " your help"
    ctx, cont = tok.encode(context), tok.encode(continuation)
    whole_once = tok.encode(context + continuation)
    # The SuperBPE tokenizer merges across the boundary when the pair is encoded whole.
    assert whole_once[:len(ctx)] != ctx

    whole, scored = encode_pair(tok.encode, EOD_TOKEN_ID, context, continuation)
    assert whole == [EOD_TOKEN_ID] + ctx + cont
    assert scored == cont
    assert tok.decode(scored) == continuation
    assert tok.decode(whole[1:]) == context + continuation


def test_trailing_context_whitespace_moves_to_the_continuation():
    tok = load_tokenizer()
    encode_pair = _encode_pair()
    assert (encode_pair(tok.encode, EOD_TOKEN_ID, "Thank you for ", "your help")
            == encode_pair(tok.encode, EOD_TOKEN_ID, "Thank you for", " your help"))


def test_pair_without_merge_matches_whole_encoding():
    tok = load_tokenizer()
    encode_pair = _encode_pair()
    context, continuation = "The answer is", " 42."
    whole_once = tok.encode(context + continuation)
    ctx = tok.encode(context)
    assert whole_once[:len(ctx)] == ctx
    whole, scored = encode_pair(tok.encode, EOD_TOKEN_ID, context, continuation)
    assert whole == [EOD_TOKEN_ID] + whole_once
    assert scored == whole_once[len(ctx):]


def test_empty_continuation_is_refused():
    tok = load_tokenizer()
    with pytest.raises(ValueError, match="empty continuation"):
        _encode_pair()(tok.encode, EOD_TOKEN_ID, "Thank you", "")
