"""Frozen thematic panels: reference semantics, drift refusal, and trainer wiring."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from kefir.trainer.cli import _validation_metric, main
from kefir.trainer.thematic_validation import (
    SEMANTIC_VOCAB_SIZE,
    ThematicPanels,
    evaluate_thematic,
    load_thematic_panels,
)
from kefir.trainer.wandb_logging import WandbTelemetry

CONTEXT = 8
VOCAB = 16
PADDED_VOCAB = 24


class _ToyCausalModel(nn.Module):
    """Tiny strictly causal model: a prefix mean of embeddings, then a wide projection.

    Float64 throughout, so that batched right-padded windows and the reference's
    cropped one-document forwards agree to far better than the asserted tolerance,
    and the projection is deliberately wider than the semantic vocabulary so the
    narrowing step is exercised.
    """

    def __init__(self, width: int = PADDED_VOCAB) -> None:
        super().__init__()
        torch.manual_seed(17)
        self.embedding = nn.Embedding(VOCAB, 6)
        self.projection = nn.Linear(6, width)
        self.double()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        positions = torch.arange(
            1, embedded.shape[1] + 1, device=embedded.device, dtype=embedded.dtype
        )
        prefix_mean = embedded.cumsum(dim=1) / positions.reshape(1, -1, 1)
        return self.projection(prefix_mean)


def _window(document: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = len(document) - 1
    if length < 1 or length > CONTEXT:
        raise ValueError("test document does not fit one window")
    inputs = np.zeros(CONTEXT, dtype=np.uint16)
    targets = np.zeros(CONTEXT, dtype=np.uint16)
    mask = np.zeros(CONTEXT, dtype=np.uint8)
    inputs[:length] = document[:length]
    targets[:length] = document[1 : length + 1]
    mask[:length] = 1
    return inputs, targets, mask


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_panels(
    root: Path,
    buckets: dict[str, list[list[int]]],
    *,
    poison_padding: bool = False,
) -> Path:
    """Write a two-bucket panel directory shaped exactly like the frozen v1 panels."""

    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "version": "test-panels",
        "context": CONTEXT,
        "tokenizer_sha256": "0" * 64,
        "buckets": {},
    }
    for name, documents in buckets.items():
        windows = [_window(document) for document in documents]
        input_ids = np.stack([window[0] for window in windows])
        target_ids = np.stack([window[1] for window in windows])
        mask = np.stack([window[2] for window in windows])
        if poison_padding:
            # Masked-out positions must never reach the loss: give them tokens whose
            # NLL is large and whose targets disagree with the causal shift.
            input_ids[mask == 0] = VOCAB - 1
            target_ids[mask == 0] = VOCAB - 2
        path = root / f"{name}.npz"
        np.savez(path, input_ids=input_ids, target_ids=target_ids, mask=mask)
        manifest["buckets"][name] = {  # type: ignore[index]
            "sha256": _digest(path),
            "target_tokens": int(mask.sum()),
            "documents": [f"{name}-{index}" for index in range(len(documents))],
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


DOCUMENTS = {
    "alpha": [[3, 5, 7, 2, 9, 1, 4, 8, 6], [2, 4, 6], [11, 1, 13, 5]],
    "beta": [[9, 8, 7, 6, 5], [1, 2, 3, 4, 5, 6, 7]],
}


def _reference_bucket_loss(model: nn.Module, path: Path) -> tuple[float, int, list[float]]:
    """The reference evaluator's semantics: one cropped document per forward."""

    with np.load(path) as arrays:
        input_ids, target_ids, mask = arrays["input_ids"], arrays["target_ids"], arrays["mask"]
    total, count, means = 0.0, 0, []
    with torch.inference_mode():
        for window, targets, window_mask in zip(input_ids, target_ids, mask, strict=True):
            length = int(window_mask.sum())
            inputs = torch.from_numpy(window[:length].astype(np.int64)).unsqueeze(0)
            labels = torch.from_numpy(targets[:length].astype(np.int64)).unsqueeze(0)
            logits = model(inputs).narrow(-1, 0, VOCAB)
            losses = F.cross_entropy(
                logits.reshape(-1, VOCAB).float(), labels.reshape(-1), reduction="none"
            )
            window_nll = float(losses.double().sum())
            total += window_nll
            count += length
            means.append(window_nll / length)
    return total, count, means


class ThematicPanelEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.model = _ToyCausalModel()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _evaluate(self, directory: Path, *, microbatch: int = 3) -> dict[str, object]:
        return evaluate_thematic(
            self.model,
            device=torch.device("cpu"),
            microbatch=microbatch,
            panels=load_thematic_panels(directory),
            semantic_vocab_size=VOCAB,
        )

    def test_batched_panel_loss_matches_the_document_by_document_reference(self) -> None:
        directory = _write_panels(self.root / "panels", DOCUMENTS)
        result = self._evaluate(directory)

        total_nll, total_targets = 0.0, 0
        for name in DOCUMENTS:
            nll, targets, means = _reference_bucket_loss(self.model, directory / f"{name}.npz")
            total_nll += nll
            total_targets += targets
            bucket = result[name]
            self.assertEqual(bucket["target_tokens"], targets)
            self.assertAlmostEqual(bucket["loss"], nll / targets, delta=1e-6)
            # The documents have unequal lengths, so a mean of window means is a
            # different number: this asserts the token weighting, not just the value.
            self.assertGreater(abs(bucket["loss"] - sum(means) / len(means)), 1e-6)

        self.assertEqual(sorted(result), ["alpha", "beta", "elapsed_seconds", "weighted_loss"])
        self.assertAlmostEqual(result["weighted_loss"], total_nll / total_targets, delta=1e-6)
        self.assertEqual(
            sum(int(result[name]["target_tokens"]) for name in DOCUMENTS), total_targets
        )
        self.assertGreater(result["elapsed_seconds"], 0.0)

    def test_right_padding_never_enters_the_loss(self) -> None:
        clean = self._evaluate(_write_panels(self.root / "clean", DOCUMENTS))
        poisoned = self._evaluate(
            _write_panels(self.root / "poisoned", DOCUMENTS, poison_padding=True)
        )
        for name in DOCUMENTS:
            self.assertEqual(poisoned[name]["target_tokens"], clean[name]["target_tokens"])
            self.assertAlmostEqual(poisoned[name]["loss"], clean[name]["loss"], delta=1e-9)
        self.assertAlmostEqual(poisoned["weighted_loss"], clean["weighted_loss"], delta=1e-9)

    def test_the_microbatch_split_does_not_change_the_result(self) -> None:
        directory = _write_panels(self.root / "panels", DOCUMENTS)
        one = self._evaluate(directory, microbatch=1)
        many = self._evaluate(directory, microbatch=32)
        for name in DOCUMENTS:
            self.assertAlmostEqual(one[name]["loss"], many[name]["loss"], delta=1e-9)

    def test_the_model_is_left_exactly_as_it_was_found(self) -> None:
        directory = _write_panels(self.root / "panels", DOCUMENTS)
        before = [parameter.detach().clone() for parameter in self.model.parameters()]
        self.model.train()
        self._evaluate(directory)
        self.assertTrue(self.model.training)
        for previous, parameter in zip(before, self.model.parameters(), strict=True):
            self.assertTrue(torch.equal(previous, parameter))

    def test_a_drifted_panel_file_is_refused(self) -> None:
        directory = _write_panels(self.root / "drifted", DOCUMENTS)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        manifest["buckets"]["alpha"]["sha256"] = "f" * 64
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError) as error:
            load_thematic_panels(directory)
        self.assertIn("drifted from the manifest", str(error.exception))

    def test_a_rewritten_panel_payload_is_refused(self) -> None:
        directory = _write_panels(self.root / "rewritten", DOCUMENTS)
        with np.load(directory / "alpha.npz") as arrays:
            payload = {key: np.array(arrays[key]) for key in arrays.files}
        payload["input_ids"][0, 0] = (int(payload["input_ids"][0, 0]) + 1) % VOCAB
        np.savez(directory / "alpha.npz", **payload)
        with self.assertRaises(ValueError):
            load_thematic_panels(directory)

    def test_a_declared_target_count_that_disagrees_is_refused(self) -> None:
        directory = _write_panels(self.root / "miscounted", DOCUMENTS)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        manifest["buckets"]["beta"]["target_tokens"] = 1
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(ValueError) as error:
            load_thematic_panels(directory)
        self.assertIn("valid targets", str(error.exception))

    def test_panels_are_loaded_once_per_directory(self) -> None:
        directory = _write_panels(self.root / "cached", DOCUMENTS)
        first = load_thematic_panels(directory)
        self.assertIs(first, load_thematic_panels(str(directory)))


class FrozenPanelsInRepositoryTests(unittest.TestCase):
    PANELS = Path(__file__).resolve().parents[1] / "src" / "kefir" / "assets" / "thematic_panels_v1"

    def test_the_repository_carries_the_verified_v1_panels(self) -> None:
        panels = load_thematic_panels(self.PANELS)
        self.assertIsInstance(panels, ThematicPanels)
        self.assertEqual(len(panels.buckets), 12)
        self.assertEqual(panels.context, 2048)
        self.assertEqual(panels.window_count, 384)
        self.assertEqual(panels.target_tokens, 256_317)
        self.assertEqual(
            panels.tokenizer_sha256,
            "068e20ef8a16b1bff2a2a1d161e5a6f1cfd0935befa7c0ae734c9ec376d373bb",
        )



def _validation_events(output: str) -> list[dict[str, object]]:
    events = []
    for line in output.splitlines():
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and "validation" in payload:
            events.append(payload["validation"])
    return events


class _Metrics:
    def __init__(self, step: int) -> None:
        self.step = step

    def as_dict(self) -> dict[str, object]:
        return {"step": self.step, "loss": 1.0}


class _Engine:
    def __init__(self, model: nn.Module) -> None:
        self.global_step = 0
        self.model = model
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(precision="fp32")

    def evaluate(self, batches, *, maximum_batches=None):
        list(batches)
        return {"loss": 1.25, "perplexity": 3.5, "target_tokens": 4, "blocks": 1}


class _Session:
    def __init__(self, engine: _Engine) -> None:
        self.engine = engine
        self.saved_validation: list[object] = []

    def step(self):
        self.engine.global_step += 1
        return _Metrics(self.engine.global_step)

    def save_checkpoint(self, coordinator, checkpoint_id: str, **kwargs):
        self.saved_validation.append(kwargs.get("validation_metrics"))
        return f"/tmp/{checkpoint_id}"


class _ValidationReader:
    def iter_from_start(self, maximum_blocks=None):
        return iter((object(),))


class _Run:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []
        self.exit_code: int | None = None

    def log(self, payload: dict[str, object]) -> None:
        self.payloads.append(dict(payload))

    def finish(self, *, exit_code: int) -> None:
        self.exit_code = exit_code


class ThematicTrainerWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_trainer(self, argv: list[str]) -> tuple[_Session, _Run, str]:
        engine = _Engine(_ToyCausalModel(width=SEMANTIC_VOCAB_SIZE + 8))
        session = _Session(engine)
        trainer_config = SimpleNamespace(evaluation_every_steps=1, checkpoint_every_steps=0)
        run = _Run()
        stream = io.StringIO()
        with (
            patch(
                "kefir.trainer.cli.setup",
                return_value=(object(), trainer_config, engine, session, object()),
            ),
            patch("kefir.trainer.cli.validation_reader", return_value=_ValidationReader()),
            patch("kefir.trainer.cli.configure_remote_publication", return_value=None),
            patch("kefir.trainer.cli.configure_wandb", return_value=WandbTelemetry(run=run)),
            patch("kefir.trainer.cli.torch.cuda.is_available", return_value=False),
            contextlib.redirect_stdout(stream),
        ):
            self.assertEqual(main(argv), 0)
        return session, run, stream.getvalue()

    def _argv(self, *extra: str) -> list[str]:
        return [
            "--dataset-dir", "/tmp/data",
            "--checkpoint-dir", str(self.root / "checkpoints"),
            "--steps", "1",
            "--evaluation-every-steps", "1",
            "--validation-blocks", "1",
            *extra,
        ]

    def test_thematic_losses_are_nested_under_the_scalar_validation_loss(self) -> None:
        directory = _write_panels(self.root / "panels", DOCUMENTS)
        session, run, output = self._run_trainer(
            self._argv("--thematic-panels", str(directory), "--thematic-microbatch", "2")
        )

        events = _validation_events(output)
        self.assertEqual(len(events), 1)
        validation = events[0]
        self.assertEqual(validation["loss"], 1.25)
        self.assertIsInstance(validation["loss"], float)
        self.assertEqual(sorted(validation["thematic"]),
                         ["alpha", "beta", "elapsed_seconds", "weighted_loss"])
        self.assertEqual(validation["thematic"]["alpha"]["target_tokens"], 13)
        self.assertEqual(validation["thematic"]["beta"]["target_tokens"], 10)
        # Best-checkpoint selection still sees a scalar held-out loss at the top level.
        self.assertEqual(_validation_metric(validation), -1.25)

        logged = [payload for payload in run.payloads if "validation/loss" in payload]
        self.assertEqual(len(logged), 1)
        payload = logged[0]
        self.assertEqual(payload["validation/loss"], 1.25)
        for name in DOCUMENTS:
            self.assertIn(f"validation/thematic/{name}/loss", payload)
            self.assertIn(f"validation/thematic/{name}/target_tokens", payload)
        self.assertIn("validation/thematic/weighted_loss", payload)
        self.assertNotIn("validation/thematic", payload)
        self.assertEqual(
            [metrics["loss"] for metrics in session.saved_validation if metrics], [1.25]
        )

    def test_the_default_run_computes_no_thematic_losses(self) -> None:
        _, run, output = self._run_trainer(self._argv())
        events = _validation_events(output)
        self.assertEqual(len(events), 1)
        self.assertNotIn("thematic", events[0])
        payload = next(item for item in run.payloads if "validation/loss" in item)
        self.assertFalse([key for key in payload if key.startswith("validation/thematic")])


if __name__ == "__main__":
    unittest.main()
