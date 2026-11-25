"""SFT contracts and equivalence with the code that trained Kefir 1.

Golden values marked RAN were produced by running the private run-time code (the
batch-screen training worker, its warmup/decay workers and the frozen evaluator) on these
same synthetic inputs; docs/code-record.md records how. Each test fails if the public
path drifts from that behavior.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from kefir.sft import data as sft_data
from kefir.sft.recipe import Execution, Recipe
from kefir.sft.renderer import EOD, MSG_END, MSG_START, ChatRenderer, ContextOverflow
from kefir.sft.train import classify_start, train
from kefir.trainer.config import TrainerConfig
from kefir.trainer.schedule import TokenLRScheduler
from kefir.trainer.state import load_trainer_state_file
from tests import sft_fixtures as fx

RENDERER_C_SHA256 = "d67c871fa0e5e3e7578ddb34ffdd99cc078c1d2448dd8f5babd040315a5b99f1"
# RAN: token-row files rendered by the run-time renderer from fx.conversation(0..6), (7..11), (12..13).
GOLDEN_ROWS_SHA256 = ["5cfe10db7019a47b5a7e5f6b5b1a3fa5588e8b407cf125ce7a0257b4d02e2286",
                      "b59c5c1e0c88b241f69f7c39895dfc4dff57861a81843b06a4516f1cad5d73e8",
                      "3efaac0e8b927b3652725724233af47831ce4d8c16ad45df17019d99bb261389"]
# RAN: every (inputs, labels) tensor of the 5 updates built by the training worker code.
GOLDEN_BATCH_SHA256 = "91db6fba11eff580d1dade7dff1d454a8fbbfa70baaf2235db3889ca65dc7c3f"
# RAN: warmup-worker spans (warm 2 updates = 6 chats, stable end at 8 chats, total).
GOLDEN_SPANS = [112, 136, 218]
# RAN: per-update LR of the warmup arms (shared prefix then load_fork_state_dict) and of
# the late decay fork from the 7-chat flat endpoint.
GOLDEN_LR = {
    "warmup-decay": [0.0009285714285714287, 0.002, 0.001985807520152837, 0.0003535348830948184, 0.0002],
    "warmup-stable-decay": [0.0009285714285714287, 0.002, 0.002, 0.00045153056575929175, 0.0002],
    "late-decay": [0.00037188470506254736, 0.0002],
}
# RAN: per-update loss of the flat run (CPU fp32, one thread).
GOLDEN_FLAT_LOSS = [17.388837814331055, 17.13192367553711, 15.931716918945312,
                    17.60847282409668, 17.020145416259766]


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("sft-inputs")
    return root, fx.write_inputs(root)


def run(root: Path, inputs: dict, name: str, *, schedule=None, resume=None, observe=(6, 7), **changes):
    value = {**fx.recipe(inputs, schedule=schedule, observe=list(observe)), **changes}
    recipe, base = Recipe.from_json(fx.write_recipe(root, value, name + ".json"))
    out = root / "runs" / name
    result = train(recipe, base, Execution(output=str(out), device="cpu", threads=1), resume=resume)
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    return result, out, events


def final_model(out: Path) -> dict:
    return load_trainer_state_file(out / "exposure-00012" / "trainer_state.pt")["model"]


def same(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def train_lrs(events) -> list[float]:
    return [e["metrics"]["learning_rate"] for e in events if e["kind"] == "train"]


# Renderer C ----------------------------------------------------------------------------

def test_renderer_is_the_renderer_c_file_that_ran():
    path = Path(__import__("kefir.sft.renderer", fromlist=["x"]).__file__)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == RENDERER_C_SHA256
    identity = ChatRenderer.load().identity
    assert identity["renderer_revision"] == "kefir-chat-c-v1"
    assert identity["renderer_sha256"] == RENDERER_C_SHA256
    assert (MSG_START, MSG_END, EOD) == (7993, 7994, 7992)


def test_renderer_reserved_ids_and_assistant_only_mask():
    renderer = ChatRenderer.load()
    chat = fx.conversation(3)["messages"]  # System + 1 exchange
    out = renderer.render(chat)
    ids, mask = out["token_ids"], out["target_mask"]
    expected_ids, expected_mask = [], []
    for message in chat:
        header = [MSG_START] + renderer.encode_content(message["role"] + "\n")
        body = renderer.encode_content(message["content"])
        assistant = message["role"] == "assistant"
        expected_ids += header + body + [EOD if assistant else MSG_END]
        expected_mask += [False] * len(header) + [assistant] * (len(body) + 1)
    assert (ids, mask) == (expected_ids, expected_mask)
    # Only Assistant text and its EOD are targets; reserved IDs are never ordinary text.
    assert all(token < EOD for token, keep in zip(ids, mask) if keep and token != EOD)
    assert ids.count(EOD) == 1 and ids[-1] == EOD and mask[-1]
    prompt = renderer.generation_prompt(chat[:-1])
    assert prompt == ids[:len(ids) - len(renderer.encode_content(chat[-1]["content"])) - 1]
    assert prompt[-len(renderer.encode_content("assistant\n")) - 1] == MSG_START


def test_whole_conversations_only_no_truncation():
    renderer = ChatRenderer.load()
    long = [{"role": "user", "content": "word " * 2100}, {"role": "assistant", "content": "ok"}]
    with pytest.raises(ContextOverflow):
        renderer.render(long)
    with pytest.raises(ValueError, match="alternating"):
        renderer.render([{"role": "assistant", "content": "hi"}])
    with pytest.raises(ValueError, match="final Assistant"):
        renderer.render([{"role": "user", "content": "hi"}])


def test_rendered_rows_match_the_run_time_renderer(inputs):
    root, spec = inputs
    hashes = [s["sha256"] for s in spec["segments"]] + [spec["validation"]["sha256"]]
    assert hashes == GOLDEN_ROWS_SHA256


@pytest.mark.parametrize("change, message", [
    (lambda r: r.update(input_ids=r["input_ids"][:-1], assistant_mask=r["assistant_mask"][:-1]), "end with"),
    # A supervised reserved control (here a User/System end marker) is rejected; masks on
    # ordinary header text are not detectable here: Renderer C is the authority for them.
    (lambda r: r["assistant_mask"].__setitem__(r["input_ids"].index(7994), True), "mask"),
    (lambda r: r["assistant_mask"].__setitem__(0, True), "mask"),
    (lambda r: r.update(input_ids=r["input_ids"] * 200, assistant_mask=r["assistant_mask"] * 200), "length"),
])
def test_read_rows_rejects_invalid_rows(tmp_path, change, message):
    fx.write_rows(tmp_path / "rows.jsonl", range(2))
    rows = [json.loads(x) for x in (tmp_path / "rows.jsonl").read_text().splitlines()]
    change(rows[0])
    (tmp_path / "bad.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match=message):
        sft_data.read_rows(tmp_path / "bad.jsonl")


# Stream, batches and the committed-target clock ----------------------------------------

def stream_of(root: Path, spec: dict) -> sft_data.Stream:
    segments = [(sft_data.read_rows(root / s["path"]), json.loads((root / s["order"]["path"]).read_text()))
                for s in spec["segments"]]
    return sft_data.Stream.build(segments, batch_size=3, seed=17)


def test_batches_match_the_run_time_geometry(inputs):
    root, spec = inputs
    stream = stream_of(root, spec)
    assert [len(b) for b in stream.batches] == [3, 3, 1, 3, 2]  # each segment batched on its own
    digest = hashlib.sha256()
    for step, batch in enumerate(stream.batches):
        rows = [stream.rows[i] for i in batch]
        tokens = sft_data.batch_tensors(rows, step)
        assert tokens.input_ids.shape == (len(rows), 2048)
        for i, row in enumerate(rows):
            n = len(row["input_ids"]) - 1
            assert torch.all(tokens.input_ids[i, n:] == EOD) and torch.all(tokens.labels[i, n:] == -100)
        assert tokens.target_token_count == sum(sft_data.target_count(r) for r in rows)
        digest.update(tokens.input_ids.numpy().tobytes())
        digest.update(tokens.labels.numpy().tobytes())
    assert digest.hexdigest() == GOLDEN_BATCH_SHA256


def test_unordered_segment_uses_the_seeded_shuffle(inputs):
    import random
    root, spec = inputs
    rows = sft_data.read_rows(root / spec["segments"][0]["path"])
    order = list(range(len(rows)))
    random.Random(18).shuffle(order)
    stream = sft_data.Stream.build([(rows, None)], batch_size=3, seed=18)
    assert [list(b) for b in stream.batches] == [order[i:i + 3] for i in range(0, len(order), 3)]


def lr_sequence(stream: sft_data.Stream, config: TrainerConfig, start: int = 0) -> list[float]:
    scheduler = TokenLRScheduler(torch.optim.SGD([torch.nn.Parameter(torch.zeros(()))], lr=1.0), config)
    return [scheduler._lr_at(stream.targets_after(step + 1)) for step in range(start, len(stream.batches))]


def test_chat_schedule_normalizes_onto_the_committed_target_clock(inputs):
    root, spec = inputs
    stream = stream_of(root, spec)

    def config(schedule):
        return Recipe(**fx.recipe(spec, schedule=schedule)).trainer_config(stream, microbatch_size=2)
    wsd = config({"kind": "wsd", "warmup_chats": 6, "decay_start_chats": 8, "decay_end_chats": 12,
                  "minimum_lr_ratio": 0.1})
    assert [wsd.warmup_tokens, wsd.warmup_tokens + wsd.stable_tokens,
            wsd.warmup_tokens + wsd.stable_tokens + wsd.decay_tokens] == GOLDEN_SPANS
    assert lr_sequence(stream, wsd) == GOLDEN_LR["warmup-stable-decay"]
    immediate = config({"kind": "wsd", "warmup_chats": 6, "decay_start_chats": 6, "decay_end_chats": 12})
    assert lr_sequence(stream, immediate) == GOLDEN_LR["warmup-decay"]
    late = config({"kind": "wsd", "warmup_chats": 0, "decay_start_chats": 7, "decay_end_chats": 12})
    assert lr_sequence(stream, late, start=3) == GOLDEN_LR["late-decay"]
    flat = config({"kind": "constant"})
    assert set(lr_sequence(stream, flat)) == {2e-3}
    with pytest.raises(ValueError, match="wsd needs"):
        config({"kind": "wsd", "warmup_chats": 9, "decay_start_chats": 8, "decay_end_chats": 12})
    with pytest.raises(ValueError, match="schedule must be"):
        config({"kind": "cosine"})


def test_start_classification_is_total():
    base = TrainerConfig(optimizer="hybrid_muon_adamw", precision="fp32").as_dict()
    assert classify_start(base, dict(base)) == "resume"
    assert classify_start(base, {**base, "microbatch_size": 4}) == "resume"
    assert classify_start(base, {**base, "schedule": "wsd", "decay_tokens": 5}) == "fork"
    assert classify_start(base, {**base, "weight_decay": 0.0}) == "fork"  # the core fork loader rejects it


# Loss ----------------------------------------------------------------------------------

def test_loss_is_the_global_mean_over_assistant_targets(inputs):
    from kefir.artifact import load_weights
    from kefir.model.engine import MoETrainerEngine
    root, spec = inputs
    stream = stream_of(root, spec)
    model, _, _ = load_weights(root / spec["parent"]["path"])
    model.train()
    batch = sft_data.batch_tensors([stream.rows[i] for i in stream.batches[0]], 0)
    with torch.no_grad():
        logits = model(batch.input_ids)
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(), batch.labels.reshape(-1),
        ignore_index=-100, reduction="sum") / batch.target_token_count
    engine = MoETrainerEngine(model, TrainerConfig(optimizer="hybrid_muon_adamw", precision="fp32",
                                                   learning_rate=2e-3, microbatch_size=2),
                              device="cpu", optimizer_telemetry=False)
    metrics = engine.train_batch(batch)
    assert metrics.target_tokens == batch.target_token_count
    assert metrics.loss == pytest.approx(float(expected), rel=1e-5)


# Training loop: fresh, exact resume, forks ---------------------------------------------

@pytest.fixture(scope="module")
def flat(inputs):
    root, spec = inputs
    return run(root, spec, "flat")


def test_flat_run_reproduces_the_run_time_trajectory(flat):
    result, out, events = flat
    assert result["mode"] == "fresh" and result["steps"] == 5 and result["chats"] == 12
    losses = [e["metrics"]["loss"] for e in events if e["kind"] == "train"]
    assert losses == pytest.approx(GOLDEN_FLAT_LOSS, rel=1e-4)
    assert set(train_lrs(events)) == {2e-3}
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "setup" and kinds[-1] == "complete"
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["exposure-00006", "exposure-00007", "exposure-00012"]
    receipt = json.loads((out / "exposure-00012" / "receipt.json").read_text())
    assert receipt["assistant_tokens_seen"] == GOLDEN_SPANS[2] and receipt["step"] == 5
    assert receipt["checkpoint_sha256"] == hashlib.sha256((out / "exposure-00012" / "trainer_state.pt").read_bytes()).hexdigest()


def test_observation_does_not_perturb_training(inputs, flat):
    root, spec = inputs
    _, quiet, _ = run(root, spec, "flat-unobserved", observe=())
    assert same(final_model(quiet), final_model(flat[1]))


def test_exact_resume_equals_uninterrupted(inputs, flat):
    root, spec = inputs
    result, out, events = run(root, spec, "flat-resumed", resume=flat[1] / "exposure-00007")
    assert result["mode"] == "resume" and events[0]["kind"] == "resume_verified"
    assert any(e["kind"] == "resume_validation_verified" for e in events)
    assert same(final_model(out), final_model(flat[1]))


def test_late_decay_fork_matches_the_run_time_fork(inputs, flat):
    root, spec = inputs
    result, _, events = run(root, spec, "late-decay", resume=flat[1] / "exposure-00007",
                            schedule={"kind": "wsd", "warmup_chats": 0, "decay_start_chats": 7,
                                      "decay_end_chats": 12, "minimum_lr_ratio": 0.1})
    assert result["mode"] == "fork" and events[0]["kind"] == "fork_verified"
    assert train_lrs(events) == GOLDEN_LR["late-decay"]
    assert math.isclose(result["final_learning_rate"], 2e-4, rel_tol=1e-12)


def test_fork_after_shared_warmup_equals_the_straight_arm(inputs, flat):
    """The warmup arms ran as shared prefix + load_fork_state_dict; straight is identical."""
    root, spec = inputs
    stable = {"kind": "wsd", "warmup_chats": 6, "decay_start_chats": 8, "decay_end_chats": 12}
    immediate = {"kind": "wsd", "warmup_chats": 6, "decay_start_chats": 6, "decay_end_chats": 12}
    _, prefix, _ = run(root, spec, "arm-prefix", schedule=stable)
    result, forked, events = run(root, spec, "arm-forked", schedule=immediate,
                                 resume=prefix / "exposure-00006")
    _, straight, straight_events = run(root, spec, "arm-straight", schedule=immediate)
    assert result["mode"] == "fork"
    assert same(final_model(forked), final_model(straight))
    assert not same(final_model(straight), final_model(flat[1]))  # positive control: the comparison discriminates
    assert train_lrs(straight_events) == GOLDEN_LR["warmup-decay"]
    assert train_lrs(events) == GOLDEN_LR["warmup-decay"][2:]


def test_resume_rejects_other_streams_tampering_and_non_schedule_forks(inputs, flat):
    root, spec = inputs
    checkpoint = flat[1] / "exposure-00007"
    other = dict(spec, segments=[spec["segments"][1], spec["segments"][0]])
    with pytest.raises(ValueError, match="prefix of this stream"):
        run(root, other, "wrong-stream", resume=checkpoint)
    with pytest.raises(ValueError, match="non-schedule"):
        run(root, spec, "wrong-fork", resume=checkpoint, optimizer={
            "optimizer": "hybrid_muon_adamw", "learning_rate": 2e-3, "weight_decay": 0.0,
            "max_grad_norm": 1.0})
    with pytest.raises(ValueError, match="continuous"):
        run(root, spec, "jump", resume=checkpoint,
            schedule={"kind": "wsd", "warmup_chats": 0, "decay_start_chats": 2, "decay_end_chats": 12})
    copy = root / "tampered"
    copy.mkdir()
    (copy / "receipt.json").write_text((checkpoint / "receipt.json").read_text())
    data = bytearray((checkpoint / "trainer_state.pt").read_bytes())
    data[-1] ^= 1
    (copy / "trainer_state.pt").write_bytes(bytes(data))
    with pytest.raises(ValueError, match="differs from its receipt"):
        run(root, spec, "tampered-run", resume=copy)


def test_recipe_rejects_unpinned_inputs_and_unknown_fields(inputs, tmp_path):
    _, spec = inputs
    with pytest.raises(ValueError, match="sha256"):
        Recipe(**{**fx.recipe(spec), "validation": {"path": "x"}})
    with pytest.raises(ValueError, match="optimizer may only set"):
        Recipe(**{**fx.recipe(spec), "optimizer": {"warmup_tokens": 5}})
    path = tmp_path / "r.json"
    path.write_text(json.dumps({**fx.recipe(spec), "extra": 1}))
    with pytest.raises(ValueError, match="unknown fields"):
        Recipe.from_json(path)


def test_two_data_parallel_ranks_equal_one_process(inputs, tmp_path):
    """The torchrun path (2 ranks, same global batch, uneven tails replicated) is exact."""
    import os
    import subprocess
    import sys
    root, spec = inputs
    path = fx.write_recipe(root, fx.recipe(spec, batch=4, observe=[8]), "two-rank.json")
    recipe, base = Recipe.from_json(path)
    single = tmp_path / "single"
    train(recipe, base, Execution(output=str(single), device="cpu", threads=1))
    env = {**os.environ, "OMP_NUM_THREADS": "1",
           "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[1] / "src"),
                                          str(Path(__file__).resolve().parents[1])])}
    subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                    str(Path(__file__).with_name("sft_two_rank_worker.py")), str(path), str(tmp_path / "two")],
                   check=True, env=env, capture_output=True, timeout=600)
    assert same(final_model(tmp_path / "two"), final_model(single))
