"""Network-free tests for concurrent 10B shard production and consumption."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from pathlib import Path
from unittest.mock import patch

from kefir.data.incremental_frontier import (
    FrontierShard,
    IncrementalRollingShardCache as BaseIncrementalRollingShardCache,
    RUN_CONTRACT_FILENAME,
    SHARD_FRONTIER_FILENAME,
    _download_verified,
    build_consumer_manifest,
    build_run_contract,
    publish_frontier,
    publish_run_contract,
    stage_incremental_window,
)
from kefir.data.incremental_cache import IncrementalRollingShardCache
from kefir.data.incremental_stage import stage_incremental_window_when_ready


class FakeStore:
    def __init__(self) -> None:
        self.bucket_id = "fake-bucket"
        self.json_objects: dict[str, dict[str, object]] = {}
        self.blobs: dict[str, bytes] = {}
        self.downloads: list[str] = []

    @staticmethod
    def object_key(run_id: str, logical_name: str) -> str:
        return f"run/{run_id}/{logical_name}"

    def _write_json(self, key: str, payload: dict[str, object]) -> None:
        self.json_objects[key] = copy.deepcopy(payload)

    def _read_json(self, key: str) -> dict[str, object] | None:
        payload = self.json_objects.get(key)
        return copy.deepcopy(payload) if payload is not None else None

    def download_shard(
        self,
        *,
        run_id: str,
        logical_name: str,
        file_id: str,
        destination: Path,
        byte_size: int,
        sha256: str,
    ) -> None:
        expected = self.object_key(run_id, logical_name)
        if file_id != expected:
            raise AssertionError("unexpected fake remote object identity")
        payload = self.blobs[expected]
        if len(payload) != byte_size or hashlib.sha256(payload).hexdigest() != sha256:
            raise AssertionError("fake remote payload identity mismatch")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        self.downloads.append(logical_name)


def _contract(*, validation_blocks: int = 1) -> dict[str, object]:
    # 4 context tokens * 2 sequences = 8 target tokens/update. 800 tokens gives
    # 100 updates, comfortably above the 16-update minimum warmup used by the
    # production WSD contract while keeping this fixture tiny.
    return build_run_contract(
        run_id="dataset-001",
        nominal_training_tokens=800,
        target_source_tokens=100,
        minimum_source_tokens=90,
        maximum_source_tokens=110,
        checkpoint_source_tokens=20,
        context_length=4,
        sequences_per_block=2,
        target_shard_bytes=1024,
        configuration_hash="a" * 64,
        schema_hash="b" * 64,
        work_plan_hash="c" * 64,
        validation_blocks=validation_blocks,
    )


def _entry(split: str, index: int, first: int, last: int, payload: bytes) -> dict[str, object]:
    filename = f"{split}/{split}-{index:06d}.bin"
    return {
        "filename": filename,
        "split": split,
        "byte_size": len(payload),
        "token_count": len(payload) // 2,
        "sequence_count": (last - first + 1) * 2,
        "checksum": hashlib.sha256(payload).hexdigest(),
        "first_block_id": first,
        "last_block_id": last,
        "context_length": 4,
        "int_type": "uint16",
        "byte_order": "little",
        "cumulative_cluster_source_tokens": {},
        "shard_cluster_source_tokens": {},
        "local_sha256": hashlib.sha256(payload).hexdigest(),
        "drive_file_id": f"object:{filename}",
        "remote_durable": True,
    }


def _durability(*rows: dict[str, object]) -> dict[str, object]:
    return {"version": 1, "run_id": "dataset-001", "shards": list(rows)}


def _ready_store() -> tuple[FakeStore, dict[str, object], list[dict[str, object]]]:
    store = FakeStore()
    contract = _contract()
    publish_run_contract(store, run_id="dataset-001", contract=contract)
    payloads = [b"a" * 16, b"b" * 16, b"v" * 16]
    rows = [
        _entry("train", 0, 0, 0, payloads[0]),
        _entry("train", 1, 1, 1, payloads[1]),
        _entry("validation", 0, 0, 0, payloads[2]),
    ]
    for row, payload in zip(rows, payloads, strict=True):
        store.blobs[store.object_key("dataset-001", str(row["filename"]))] = payload
    publish_frontier(
        store,
        run_id="dataset-001",
        contract=contract,
        durability_manifest=_durability(*rows),
    )
    return store, contract, rows


class IncrementalFrontierTests(unittest.TestCase):
    def test_acknowledgement_does_not_wait_for_next_ready_shard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, contract, rows = _ready_store()
            cache = IncrementalRollingShardCache(
                root=Path(tmp), run_id="dataset-001", contract=contract, store=store,
            )
            cache.planned_block_count = 2
            try:
                cache.ensure_block(0)
                current = Path(tmp) / rows[0]["filename"]
                self.assertTrue(current.is_file())
                original_wait = cache._wait_for_shard

                def only_current(block_id):
                    if block_id != 0:
                        raise AssertionError("acknowledgement waited for a future shard")
                    return original_wait(block_id)

                with patch.object(cache, "_wait_for_shard", side_effect=only_current):
                    cache.acknowledge(0)
                self.assertTrue(current.is_file())
                cache.ensure_block(1)
                self.assertFalse(current.exists())
            finally:
                cache.close()

    def test_restored_boundary_evicts_previous_shard_after_next_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, contract, rows = _ready_store()
            cache = IncrementalRollingShardCache(
                root=Path(tmp), run_id="dataset-001", contract=contract, store=store,
            )
            cache.planned_block_count = 2
            try:
                cache.ensure_block(0)
                current = Path(tmp) / rows[0]["filename"]
                cache.restore_after_acknowledged(0)
                self.assertTrue(current.is_file())
                cache.ensure_block(1)
                self.assertFalse(current.exists())
            finally:
                cache.close()

    def test_frontier_reads_and_cache_acceptance_are_one_serial_transaction(self) -> None:
        for cache_type in (BaseIncrementalRollingShardCache, IncrementalRollingShardCache):
            with self.subTest(cache_type=cache_type), tempfile.TemporaryDirectory() as tmp:
                store, contract, _ = _ready_store()
                cache = cache_type(root=Path(tmp), run_id="dataset-001", contract=contract, store=store)
                first_read = threading.Event()
                release_first = threading.Event()
                second_attempt = threading.Event()
                original_read = store._read_json
                calls = []
                lock = threading.RLock()

                class ObservedLock:
                    def __enter__(self):
                        if threading.current_thread().name == "later-reader":
                            second_attempt.set()
                        lock.acquire()

                    def __exit__(self, *args):
                        lock.release()

                cache._frontier_lock = ObservedLock()

                def delayed_read(key):
                    snapshot = original_read(key)
                    calls.append(snapshot)
                    if len(calls) == 1:
                        first_read.set()
                        if not release_first.wait(5):
                            raise TimeoutError("first frontier response was not released")
                    return snapshot

                store._read_json = delayed_read
                refresh = getattr(cache, "_refresh_ready_prefix", cache._frontier)

                def later_refresh():
                    threading.current_thread().name = "later-reader"
                    return refresh()

                try:
                    with ThreadPoolExecutor(max_workers=2) as threads:
                        first = threads.submit(refresh)
                        try:
                            self.assertTrue(first_read.wait(5))
                            key = store.object_key("dataset-001", SHARD_FRONTIER_FILENAME)
                            next_row = _entry("train", 2, 2, 2, b"c" * 16)
                            next_row = {field: next_row[field] for field in calls[0]["ready_train_shards"][0]}
                            store.json_objects[key]["ready_train_shards"].append(next_row)
                            second = threads.submit(later_refresh)
                            self.assertTrue(second_attempt.wait(5))
                            # The later request cannot overtake the response in flight.
                            self.assertEqual(len(calls), 1)
                        finally:
                            release_first.set()
                        first.result(timeout=5)
                        second.result(timeout=5)
                    self.assertEqual(len(cache._last_frontier["ready_train_shards"]), 3)
                    if isinstance(cache, IncrementalRollingShardCache):
                        self.assertEqual(cache._cached_shard(2).filename, next_row["filename"])
                finally:
                    cache.close()

    def test_serial_remote_regression_and_mutation_remain_errors(self) -> None:
        for mutation in ("regression", "checksum"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                store, contract, _ = _ready_store()
                cache = IncrementalRollingShardCache(
                    root=Path(tmp), run_id="dataset-001", contract=contract, store=store,
                )
                try:
                    cache._refresh_ready_prefix()
                    key = store.object_key("dataset-001", SHARD_FRONTIER_FILENAME)
                    if mutation == "regression":
                        store.json_objects[key]["ready_train_shards"].pop()
                    else:
                        store.json_objects[key]["ready_train_shards"][0]["checksum"] = "f" * 64
                    with self.assertRaisesRegex(RuntimeError, "regressed or mutated"):
                        cache._refresh_ready_prefix()
                    self.assertEqual(len(cache._cached_train), 2)
                    self.assertNotEqual(cache._cached_train[0].checksum, "f" * 64)
                finally:
                    cache.close()

    def test_download_rejects_unsafe_filename_before_deletion(self) -> None:
        store = FakeStore()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            root.mkdir()
            outside = Path(tmp) / "outside.bin"
            outside.write_bytes(b"sentinel")
            shard = FrontierShard(
                "../outside.bin",
                "train",
                1,
                hashlib.sha256(b"x").hexdigest(),
                0,
                0,
                1,
            )

            with self.assertRaisesRegex(RuntimeError, "unsafe local path"):
                _download_verified(
                    store,
                    run_id="dataset-001",
                    root=root,
                    shard=shard,
                )

            self.assertTrue(outside.exists())

    def test_exact_10b_contract_and_standard_wsd_are_known_before_gpu(self) -> None:
        contract = build_run_contract(
            run_id="example-10b-b64-dataset-001",
            nominal_training_tokens=10_000_000_000,
            target_source_tokens=10_000_000_000,
            minimum_source_tokens=9_000_000_000,
            maximum_source_tokens=11_000_000_000,
            checkpoint_source_tokens=500_000_000,
            context_length=2048,
            sequences_per_block=64,
            target_shard_bytes=1024**3,
            configuration_hash="a" * 64,
            schema_hash="b" * 64,
            work_plan_hash="c" * 64,
            validation_blocks=16,
        )
        trainer = contract["trainer"]
        self.assertIsInstance(trainer, dict)
        trainer = dict(trainer)
        self.assertEqual(contract["planned_train_blocks"], 76_294)
        self.assertEqual(contract["planned_train_target_tokens"], 10_000_007_168)
        self.assertEqual(trainer["warmup_updates"], 3_815)
        self.assertEqual(trainer["stable_updates"], 57_220)
        self.assertEqual(trainer["decay_updates"], 15_259)
        self.assertEqual(trainer["warmup_tokens"], 500_039_680)
        self.assertEqual(trainer["stable_tokens"], 7_499_939_840)
        self.assertEqual(trainer["decay_tokens"], 2_000_027_648)
        self.assertEqual(trainer["validation_blocks"], 16)

    def test_run_contract_is_immutable_after_first_publication(self) -> None:
        store = FakeStore()
        contract = _contract()
        publish_run_contract(store, run_id="dataset-001", contract=contract)
        mutated = dict(contract)
        mutated["planned_train_blocks"] = 5
        with self.assertRaisesRegex(RuntimeError, "refusing to mutate"):
            publish_run_contract(store, run_id="dataset-001", contract=mutated)

    def test_ready_train_frontier_is_monotonic_and_validation_freezes(self) -> None:
        store = FakeStore()
        contract = _contract()
        train0 = _entry("train", 0, 0, 0, b"a" * 16)
        val0 = _entry("validation", 0, 0, 0, b"v" * 16)
        first = publish_frontier(
            store,
            run_id="dataset-001",
            contract=contract,
            durability_manifest=_durability(train0, val0),
        )
        self.assertEqual(len(first["ready_train_shards"]), 1)
        self.assertEqual(len(first["frozen_validation_shards"]), 1)

        train1 = _entry("train", 1, 1, 1, b"b" * 16)
        val1 = _entry("validation", 1, 1, 1, b"w" * 16)
        second = publish_frontier(
            store,
            run_id="dataset-001",
            contract=contract,
            durability_manifest=_durability(train0, train1, val0, val1),
        )
        self.assertEqual(len(second["ready_train_shards"]), 2)
        self.assertEqual(second["frozen_validation_shards"], first["frozen_validation_shards"])

        mutated = dict(train0)
        mutated["checksum"] = "f" * 64
        mutated["local_sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "immutable monotonic prefix"):
            publish_frontier(
                store,
                run_id="dataset-001",
                contract=contract,
                durability_manifest=_durability(mutated, train1, val0, val1),
            )

    def test_consumer_manifest_is_stable_bootstrap_not_full_future_inventory(self) -> None:
        store, contract, _ = _ready_store()
        frontier = store._read_json(store.object_key("dataset-001", SHARD_FRONTIER_FILENAME))
        self.assertIsNotNone(frontier)
        manifest = build_consumer_manifest(contract=contract, frontier=dict(frontier))
        train = [row for row in manifest["shards"] if row["split"] == "train"]
        validation = [row for row in manifest["shards"] if row["split"] == "validation"]
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)
        self.assertEqual(manifest["incremental_frontier"]["planned_train_blocks"], 100)

    def test_cpu_stage_downloads_current_and_successor_before_gpu(self) -> None:
        store, _, rows = _ready_store()
        with tempfile.TemporaryDirectory() as tmp:
            result = stage_incremental_window(
                store=store,
                run_id="dataset-001",
                destination=Path(tmp),
                start_block_id=0,
                timeout_seconds=0.1,
                poll_seconds=0.001,
            )
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["staged_train_shards"], [rows[0]["filename"], rows[1]["filename"]])
            self.assertTrue((Path(tmp) / str(rows[0]["filename"])).is_file())
            self.assertTrue((Path(tmp) / str(rows[1]["filename"])).is_file())
            self.assertTrue((Path(tmp) / str(rows[2]["filename"])).is_file())
            self.assertTrue((Path(tmp) / RUN_CONTRACT_FILENAME).is_file())
            self.assertTrue((Path(tmp) / SHARD_FRONTIER_FILENAME).is_file())
            self.assertTrue((Path(tmp) / "manifest.json").is_file())

    def test_cpu_stage_waits_for_bootstrap_metadata_but_fails_on_integrity_error(self) -> None:
        store, contract, _ = _ready_store()
        contract_key = store.object_key("dataset-001", RUN_CONTRACT_FILENAME)
        saved_contract = store.json_objects.pop(contract_key)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TimeoutError):
                stage_incremental_window_when_ready(
                    store=store,
                    run_id="dataset-001",
                    destination=Path(tmp),
                    start_block_id=0,
                    timeout_seconds=0.01,
                    poll_seconds=0.001,
                )
        store.json_objects[contract_key] = saved_contract
        broken = copy.deepcopy(contract)
        broken["contract_sha256"] = "0" * 64
        store.json_objects[contract_key] = broken
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "hash is invalid"):
                stage_incremental_window_when_ready(
                    store=store,
                    run_id="dataset-001",
                    destination=Path(tmp),
                    start_block_id=0,
                    timeout_seconds=0.1,
                    poll_seconds=0.001,
                )

    def test_uncommitted_remote_shards_never_enter_ready_frontier(self) -> None:
        store = FakeStore()
        contract = _contract()
        train0 = _entry("train", 0, 0, 0, b"a" * 16)
        val0 = _entry("validation", 0, 0, 0, b"v" * 16)
        train1 = _entry("train", 1, 1, 1, b"b" * 16)
        train1["remote_durable"] = False
        with self.assertRaisesRegex(RuntimeError, "non-durable"):
            publish_frontier(
                store,
                run_id="dataset-001",
                contract=contract,
                durability_manifest=_durability(train0, train1, val0),
            )


if __name__ == "__main__":
    unittest.main()
