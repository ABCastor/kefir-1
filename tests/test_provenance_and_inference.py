"""What ran stays what ran: file identities, the frozen suite, the GDN-2 backend, chat."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((Path(__file__).parent / "data" / "ran-runtime-manifest.json").read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_files_are_byte_identical_to_what_ran_except_declared_changes():
    changed = MANIFEST["changed_in_public_tree"]
    drift = [name for name, digest in MANIFEST["files"].items()
             if name not in changed and sha(ROOT / name) != digest]
    assert drift == []
    # A declared change must really be a change; otherwise the declaration is stale.
    assert [name for name in changed if sha(ROOT / name) == MANIFEST["files"][name]] == []


BENCH = ROOT / "src" / "kefir" / "bench"
# The snapshot (kefir-bench/1, snapshot manifest a13c4d2d...) the SFT evaluator loaded.
SNAPSHOT = {"authored.py": "b120d29f677d6bd89d1497722184dcde90af6439ea5839125245ff2ea47d3306",
            "io.py": "49221dda1a83b2c31f81d943d245fb52da40744111f2c477368f865ee5acb620",
            "scoring.py": "7c5fdfc3d082eac8ba2a8980a2468f43740810ddbfff5afcd2f82504d09fa369",
            "public_data.py": "8683deb7510a8d90eb2ae081668627fc0ce1aed5f21c812f39ee359bd6946d55"}
DATA = {"authored-manifest.json": "3151afb7cf7184147ce9763fd112ad17270136a0ecc440bc07912e9c96ea187b",
        "source-manifest.json": "2890c65e1f40066bcb133fc610d30ac91872543669c0848acf8d7cf3697e3ae6",
        "manifest.json": "6424fa50400bef5252e2385cb23d54d732ded16de1eb78497b9ace4fb7894524"}


def test_bench_modules_are_the_snapshot_that_ran():
    for name in ("authored.py", "io.py", "scoring.py"):
        assert sha(BENCH / name) == SNAPSHOT[name], name
    # public_data.py differs only in where its data directory lives.
    source = (BENCH / "public_data.py").read_text().replace(
        '_DATA = Path(__file__).resolve().parent / "data"',
        '_DATA = Path(__file__).resolve().parent.parent / "data"')
    assert hashlib.sha256(source.encode()).hexdigest() == SNAPSHOT["public_data.py"]
    for name, digest in DATA.items():
        assert sha(BENCH / "data" / name) == digest, name


def test_authored_cases_rebuild_offline_exactly(tmp_path):
    from kefir.bench.authored import build_authored
    build_authored(tmp_path)
    assert sha(tmp_path / "authored-manifest.json") == DATA["authored-manifest.json"]
    assert {name: sha(tmp_path / f"authored-{name}.jsonl") for name in ("dev", "test", "reserve")} == {
        "dev": "19a8c352e20aecad302c276b75f25ce14ef02d806651effe9eb5d5121fe37c58",
        "test": "4e1fcb1e4f40a71061c3b94c62af2e54bc9614e87dbe70ed3ae07d3388973c54",
        "reserve": "f2eb944adff8c21b31d9994313c41e9bf69960e4c3d6a86137b595edaeed4ad0"}


def test_incomplete_suite_is_refused(tmp_path):
    from kefir.bench.authored import build_authored
    from kefir.bench.build import assemble, build_suite, load_split
    cases = build_authored(tmp_path)
    (tmp_path / "source-manifest.json").write_bytes((BENCH / "data" / "source-manifest.json").read_bytes())
    with pytest.raises(ValueError, match="frozen"):
        assemble(tmp_path, cases)  # public cases missing
    with pytest.raises(ValueError, match="frozen"):
        load_split(tmp_path, "dev")
    with pytest.raises(FileNotFoundError, match="download"):
        build_suite(tmp_path / "suite", tmp_path / "cache")


def test_pairwise_backend_matches_the_recurrent_oracle():
    from kefir.layers.gdn2 import assert_gdn2_backend_parity
    from kefir.layers.gdn2_pairwise import PairwiseGDN2Backend
    torch.manual_seed(2)
    batch, sequence, heads, key_dim, value_dim = 2, 45, 2, 4, 3
    q, k = torch.randn(batch, sequence, heads, key_dim), torch.randn(batch, sequence, heads, key_dim)
    v = torch.randn(batch, sequence, heads, value_dim)
    log_decay = -0.05 * torch.rand(batch, sequence, heads, key_dim)
    erase = torch.sigmoid(torch.randn(batch, sequence, heads, key_dim))
    write = torch.sigmoid(torch.randn(batch, sequence, heads, value_dim))
    state = torch.randn(batch, heads, key_dim, value_dim)
    assert_gdn2_backend_parity(PairwiseGDN2Backend(32), q, k, v, log_decay, erase, write, state,
                               atol=3e-5, rtol=3e-5, check_gradients=True,
                               # FP32 gradient round-off over 45 steps; against a float64 oracle the
                               # core chunkwise backend deviates more (7e-2) than this one (2e-2).
                               gradient_atol=5e-3, gradient_rtol=5e-3)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from kefir.artifact import load_weights
    from tests import sft_fixtures as fx
    torch.set_num_threads(1)
    path = fx.write_parent(tmp_path_factory.mktemp("weights") / "model.safetensors")
    return load_weights(path)[0]


def test_suite_evaluation_runs_end_to_end(tiny_model, tmp_path):
    from kefir.bench.authored import build_authored
    from kefir.sft.evaluation import evaluate_model
    from tests import sft_fixtures as fx
    cases = [c for c in build_authored(tmp_path) if c["split"] == "dev"]
    pick = [next(c for c in cases if c["category"] == name) for name in ("reasoning", "instructions", "stopping")]
    for case in pick:
        case["max_new_tokens"] = 3
    fx.write_rows(tmp_path / "rows.jsonl", range(2))
    from kefir.sft.data import read_rows
    result = evaluate_model(tiny_model, cases=pick, validation_rows=read_rows(tmp_path / "rows.jsonl"))
    assert len(result["responses"]) == 3 and result["retention"] is None
    assert result["summary"]["validation"]["conversations"] == 2
    assert all(r["termination"] in {"native_end", "output_cap"} for r in result["responses"])


def test_chat_keeps_whole_exchanges_and_streams_like_batch(tiny_model):
    from kefir.sft.chat import Chat, fit_context
    from kefir.sft.renderer import ChatRenderer
    renderer = ChatRenderer.load()
    long = [{"role": "system", "content": "Be brief."}]
    for i in range(30):
        long += [{"role": "user", "content": "tell me about rivers " * 20 + str(i)},
                 {"role": "assistant", "content": "rivers flow " * 20}]
    long.append({"role": "user", "content": "last question"})
    kept, prefix, removed = fit_context(renderer, long, 128)
    assert removed > 0 and kept[0]["role"] == "system" and kept[-1]["content"] == "last question"
    assert len(prefix) + 128 <= 2048 and prefix == renderer.generation_prompt(kept)
    chat = Chat(tiny_model, max_new_tokens=4)
    streamed = []
    first = chat.reply("Hello there", on_text=streamed.append, warn=lambda _: None)
    chat.reset()
    second = chat.reply("Hello there", warn=lambda _: None)
    assert first == second == "".join(streamed)


def test_one_command_line(tmp_path, capsys):
    from kefir.cli import main
    from tests import sft_fixtures as fx
    assert main([]) == 2
    conversations = tmp_path / "chats.jsonl"
    conversations.write_text("".join(json.dumps(fx.conversation(i)) + "\n" for i in range(7)))
    assert main(["sft", "render", "--conversations", str(conversations), "--output", str(tmp_path / "rows.jsonl")]) == 0
    assert sha(tmp_path / "rows.jsonl") == "5cfe10db7019a47b5a7e5f6b5b1a3fa5588e8b407cf125ce7a0257b4d02e2286"
    assert main(["evaluate"]) == 2
