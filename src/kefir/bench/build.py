"""Reconstruct the frozen suite (1022 cases) and prove it equals the pinned manifest.

Network access happens only with ``download=True``, from the URLs pinned in
``data/source-manifest.json``; every upstream file is checked by SHA-256. Reading the
two ARC Parquet files requires ``pyarrow``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

from kefir.staging import staged_output

from .authored import build_authored
from .io import read_jsonl, sha256, validate_cases, write_jsonl

DATA = Path(__file__).resolve().parent / "data"
SPLITS = ("dev", "test", "reserve")


def frozen_manifest() -> dict:
    return json.loads((DATA / "manifest.json").read_text())


def fetch_sources(cache: Path, *, download: bool) -> None:
    source = json.loads((DATA / "source-manifest.json").read_bytes())
    cache.mkdir(parents=True, exist_ok=True)
    for entry in source["sources"]:
        path = cache / entry["file"]
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"missing pinned source {path}; pass download=True")
            with urllib.request.urlopen(entry["url"], timeout=120) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError(f"downloaded source hash mismatch: {entry['file']}")
            with path.open("xb") as stream:
                stream.write(payload)
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"cached source hash mismatch: {entry['file']}")


def assemble(output: Path, cases: list[dict]) -> dict:
    """Write the three split files and the suite manifest exactly as the frozen builder."""
    report = validate_cases(cases)
    files = {}
    for split in SPLITS:
        rows = sorted((c for c in cases if c["split"] == split), key=lambda c: c["id"])
        write_jsonl(output / (split + ".jsonl"), rows)
        files[split + ".jsonl"] = sha256(output / (split + ".jsonl"))
    manifest = {"schema": "kefir-bench/1",
                "status": "instance-heldout diagnostic; template mechanisms partly shared",
                "files": files, "validation": report,
                "source_manifest_sha256": sha256(output / "source-manifest.json"),
                "authored_manifest_sha256": sha256(output / "authored-manifest.json"),
                "pretraining_overlap": "not checked; public benchmark exposure unknown",
                "sft_overlap": "not checked; run explicit overlap audit before recipe claims",
                "release_thresholds": None,
                "test_policy": "Confirmation and reserve are packaged; do not use for recipe tuning."}
    if manifest != frozen_manifest():
        raise ValueError("rebuilt suite differs from the frozen kefir-bench/1 manifest")
    (output / "manifest.json").write_bytes((DATA / "manifest.json").read_bytes())
    return manifest


def build_suite(output: str | Path, cache: str | Path, *, download: bool = False) -> dict:
    from importlib.util import find_spec
    if find_spec("pyarrow") is None:
        raise ModuleNotFoundError("kefir bench build reads Parquet files and needs pyarrow: install the bench "
                                  "extra (pip install -e '.[bench]' or uv sync --locked --extra bench)")
    from .public_data import select_from_sources
    cache = Path(cache)
    with staged_output(output) as work:
        work.mkdir()
        fetch_sources(cache, download=download)
        public = select_from_sources(cache, work)
        pinned = json.loads((DATA / "source-manifest.json").read_bytes())["selected_files"]
        for name, digest in pinned.items():
            if sha256(work / name) != digest:
                raise ValueError(f"public selection drift: {name}")
        authored = build_authored(work)
        return assemble(work, authored + public)


def load_split(root: str | Path, split: str = "dev") -> list[dict]:
    """Read one split of a built suite, verified against the frozen manifest hash."""
    if split not in SPLITS:
        raise ValueError("split must be dev, test or reserve")
    path = Path(root) / (split + ".jsonl")
    if sha256(path) != frozen_manifest()["files"][path.name]:
        raise ValueError(f"benchmark split differs from the frozen suite: {path}")
    cases = read_jsonl(path)
    validate_cases(cases)
    return cases


__all__ = ["assemble", "build_suite", "fetch_sources", "frozen_manifest", "load_split"]
