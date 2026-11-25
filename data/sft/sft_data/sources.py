"""The eight pinned public sources and their conversion to ``Record``.

Every source is read from files of one pinned revision. Flat chat sources map one row to
one conversation. OpenAssistant releases are message trees: each maximal valid path from a
root prompt to an Assistant reply becomes one whole conversation; shorter paths that are
prefixes of a kept path are not separate rows. A path stops at the first message that is
not English, deleted, rejected by review, or out of role order. Held-out splits are read
only to isolate their families from training rows.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

from .records import Record, RecordError, make_record, normalize_text, with_groups


@dataclass(frozen=True)
class Source:
    revision: str
    config: str
    family: str
    train: tuple
    heldout: tuple = ()
    licence: str = ""


SOURCES = {
    "HuggingFaceTB/smol-smoltalk": Source(
        "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc", "default", "smol_smoltalk",
        tuple(f"data/train-0000{i}-of-00004.parquet" for i in range(4)),
        ("data/test-00000-of-00001.parquet",), "Apache 2.0"),
    "HuggingFaceH4/ultrachat_200k": Source(
        "8049631c405ae6576f93f445c6b8166f76f5505a", "default", "prejudge_dialogue",
        ("data/train_sft-00000-of-00003-a3ecf92756993583.parquet",
         "data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet",
         "data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet"),
        ("data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet",
         "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"), "MIT"),
    "allenai/tulu-3-sft-personas-instruction-following": Source(
        "fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e", "default", "prejudge_dialogue",
        ("data/train-00000-of-00001.parquet",), (), "ODC-BY"),
    "HuggingFaceTB/smoltalk2": Source(
        "fc6cc2103c066455aade5d7fbb346039ae36ca5e", "SFT", "prejudge_dialogue",
        ("SFT/smoltalk_smollm3_systemchats_30k_no_think-00000-of-00001.parquet",), (),
        "Apache 2.0 for new parts; components keep their terms"),
    "OpenAssistant/oasst2": Source(
        "179dd21fc55192153d94adb0e0ce8f69e222bf75", "default", "openassistant",
        ("data/train-00000-of-00001-88ba0162028a73fc.parquet",),
        ("data/validation-00000-of-00001-1deeef95c3248fe0.parquet",), "Apache 2.0"),
    "databricks/databricks-dolly-15k": Source(
        "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", "default", "prejudge_dialogue",
        ("databricks-dolly-15k.jsonl",), (), "CC BY-SA 3.0"),
    "OpenAssistant/oasst1": Source(
        "fdf72ae0827c1cda404aff25b6603abec9e3399b", "default", "openassistant",
        ("data/train-00000-of-00001-b42a775f407cee45.parquet",),
        ("data/validation-00000-of-00001-134b8fd0c89408b6.parquet",), "Apache 2.0"),
    "HuggingFaceH4/no_robots": Source(
        "e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b", "default", "no_robots",
        ("data/train-00000-of-00001.parquet",), ("data/test-00000-of-00001.parquet",), "CC BY-NC 4.0"),
}
HELDOUT_SPLIT = {"openassistant": "validation"}  # otherwise "test"


def read_rows(path: Path):
    """Rows of a pinned file: JSON Lines, or Parquet (needs pyarrow)."""
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)
        return
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
        yield from batch.to_pylist()


def _pointer(relative, index):
    return {"relative_path": relative, "row_index": index}


def _dialogue_messages(row, dataset):
    if dataset == "databricks/databricks-dolly-15k":
        context = row.get("context") or ""
        user = normalize_text(row.get("instruction"))
        if not isinstance(context, str):
            raise RecordError("invalid_context")
        if context.strip():
            user += "\n\nContext:\n" + normalize_text(context)
        return [{"role": "user", "content": user}, {"role": "assistant", "content": row.get("response")}]
    messages = [dict(m) for m in row.get("messages") or []]
    if dataset == "HuggingFaceTB/smoltalk2":
        kwargs = row.get("chat_template_kwargs") or {}
        if (set(kwargs) - {"custom_instructions", "enable_thinking", "python_tools", "xml_tools"}
                or kwargs.get("enable_thinking") is not False or kwargs.get("python_tools")
                or kwargs.get("xml_tools")):
            raise RecordError("unsupported_template")
        custom = kwargs.get("custom_instructions")
        if custom is not None:
            custom = normalize_text(custom)
            if messages and messages[0].get("role") == "system":
                if messages[0].get("content") != custom:
                    raise RecordError("conflicting_system_instructions")
            else:
                messages.insert(0, {"role": "system", "content": custom})
    return messages


def dialogue_record(row, dataset, relative, index, split) -> Record:
    """One flat chat row as one conversation."""
    spec = SOURCES[dataset]
    pointer = _pointer(relative, index)
    if spec.family == "prejudge_dialogue":
        native = {k: row[k] for k in ("category", "constraints", "source", "id", "prompt_id",
                                      "chat_template_kwargs") if k in row}
        source = {"dataset": dataset, "revision": spec.revision, "config": spec.config, "split": split,
                  "row_id": f"{relative}:{index}", "family": spec.family,
                  "native_source": str(row.get("source") or row.get("category") or dataset),
                  "file_pointer": pointer}
        lineage = {"native": native, "admission": "analysis_only"}
    else:
        category = row.get("category") if spec.family == "no_robots" else row.get("source")
        if not isinstance(category, str) or not category.strip():
            raise RecordError("missing_native_category")
        row_id = f"{split}:{relative}:{index}"
        source = {"dataset": dataset, "revision": spec.revision, "config": spec.config, "split": split,
                  "row_id": row_id, "family": spec.family, "native_source": category,
                  "file_pointer": pointer}
        lineage = {"file_pointer": pointer, "native_category": category,
                   "original_row_id": row_id, "prompt_id": row.get("prompt_id")}
    return make_record(source, _dialogue_messages(row, dataset), lineage)


NATIVE_FIELDS = ("parent_id", "message_tree_id", "role", "lang", "review_result", "deleted",
                 "rank", "labels", "synthetic")


def thread_records(rows, dataset, split, pointers) -> tuple[list[Record], int]:
    """Maximal valid root-to-Assistant paths of OpenAssistant trees; returns (records, rejected)."""
    spec = SOURCES[dataset]
    nodes, children = {}, defaultdict(list)
    for row, pointer in zip(rows, pointers):
        nodes[row["message_id"]] = (row, pointer)
    for mid, (row, _) in nodes.items():
        if row.get("parent_id") is not None:
            children[row["parent_id"]].append(mid)
    paths, rejected = [], 0

    def valid(row, depth, parent):
        role = {"prompter": "user", "assistant": "assistant"}.get(row.get("role"))
        try:
            normalize_text(row.get("text"))
        except RecordError:
            return False
        return (role == ("user" if depth % 2 == 0 else "assistant") and row.get("lang") == "en"
                and row.get("deleted") is not True and row.get("review_result") is not False
                and isinstance(row.get("message_tree_id"), str)
                and (parent is None or nodes[parent][0].get("message_tree_id") == row["message_tree_id"]))

    def walk(mid, chain):
        nonlocal rejected
        row = nodes[mid][0]
        if not valid(row, len(chain), chain[-1] if chain else None):
            rejected += 1
            return
        chain = chain + [mid]
        kids = sorted(k for k in children.get(mid, ()) if k in nodes)
        if len(chain) % 2 == 0:
            paths.append(chain)
        for kid in kids:
            walk(kid, chain)

    for mid in sorted(m for m, (r, _) in nodes.items() if r.get("parent_id") is None):
        walk(mid, [])
    prefixes = {tuple(p[:i]) for p in paths for i in range(2, len(p), 2)}
    records = []
    for chain in paths:
        if tuple(chain) in prefixes:
            continue
        terminal_row, terminal_pointer = nodes[chain[-1]]
        messages = [{"role": "user" if nodes[n][0]["role"] == "prompter" else "assistant",
                     "content": nodes[n][0]["text"]} for n in chain]
        source = {"dataset": dataset, "revision": spec.revision, "config": spec.config, "split": split,
                  "row_id": chain[-1], "family": spec.family, "native_source": dataset,
                  "file_pointer": terminal_pointer}
        lineage = {"tree_id": terminal_row["message_tree_id"], "message_path": chain,
                   "terminal_message_id": chain[-1], "file_pointers": [nodes[n][1] for n in chain],
                   "native_metadata": [{"message_id": n, "file_pointer": nodes[n][1],
                                        **{k: nodes[n][0].get(k) for k in NATIVE_FIELDS}} for n in chain],
                   "review_unresolved": any(nodes[n][0].get("review_result") is None for n in chain)}
        try:
            records.append(make_record(source, messages, lineage))
        except RecordError:
            rejected += 1
    return sorted(records, key=lambda r: r.record_id), rejected


def ancestor(record: Record, keep: int) -> Record:
    """The first ``keep`` messages of a threaded conversation, keeping all its groups."""
    lineage = dict(record.lineage)
    for key in ("message_path", "native_metadata", "file_pointers"):
        lineage[key] = lineage[key][:keep]
    lineage["terminal_message_id"] = lineage["message_path"][-1]
    lineage["recovered_from"] = record.record_id
    source = dict(record.source, row_id=lineage["terminal_message_id"], file_pointer=lineage["file_pointers"][-1])
    return with_groups(make_record(source, list(record.messages[:keep]), lineage), record.group_ids)


def load(root: Path, dataset: str):
    """Yield (record or None, split) for every train and held-out row; None marks a rejected row.

    ``root/<dataset>/<relative path>`` holds the files downloaded at the pinned revision.
    """
    spec = SOURCES[dataset]
    for files, split in ((spec.train, "train"), (spec.heldout, HELDOUT_SPLIT.get(spec.family, "test"))):
        if spec.family == "openassistant":
            rows, pointers = [], []
            for relative in files:
                for index, row in enumerate(read_rows(root / dataset / relative)):
                    rows.append(row)
                    pointers.append(_pointer(relative, index))
            records, rejected = thread_records(rows, dataset, split, pointers)
            yield from ((r, split) for r in records)
            yield from ((None, split) for _ in range(rejected))
            continue
        for relative in files:
            for index, row in enumerate(read_rows(root / dataset / relative)):
                try:
                    yield dialogue_record(row, dataset, relative, index, split), split
                except RecordError:
                    yield None, split
