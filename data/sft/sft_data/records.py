"""The one conversation record and the identity facts derived from it.

A conversation enters as role/content messages plus its source pointer. It is normalised
once here (Unicode NFC, ``\\n`` line endings, strict role order); every later step reads the
same ``Record``. Identity (``record_id``, ``content_hash``) and isolation groups
(``group_ids``) are computed here and nowhere else. These hashes reproduce the identifiers
of the published labelled pool.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re
import unicodedata

SOURCE_KEY_FIELDS = ("dataset", "revision", "config", "split", "row_id")


class RecordError(ValueError):
    """A conversation that cannot be kept whole; it is rejected, never repaired."""


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def stable_hash(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_key(text: str) -> str:
    """Case- and whitespace-insensitive key for overlap checks (benchmarks, first prompts)."""
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def normalize_text(text) -> str:
    if not isinstance(text, str):
        raise RecordError("non_string_content")
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    if not text.strip():
        raise RecordError("empty_message")
    if any(0xD800 <= ord(c) <= 0xDFFF or (unicodedata.category(c) == "Cc" and c not in "\n\t")
           for c in text):
        raise RecordError("unsafe_unicode_control")
    return text


def normalize_messages(messages) -> list[dict]:
    """Optional first System message, then User/Assistant alternation, ending with Assistant."""
    if not isinstance(messages, (list, tuple)) or not messages:
        raise RecordError("missing_messages")
    result = []
    for index, item in enumerate(messages):
        if not isinstance(item, dict):
            raise RecordError("bad_message_schema")
        role = item.get("role")
        if role == "system":
            if index != 0:
                raise RecordError("misplaced_system")
        elif role in ("user", "assistant"):
            expected = "assistant" if result and result[-1]["role"] == "user" else "user"
            if role != expected:
                raise RecordError("incomplete_or_non_alternating")
        else:
            raise RecordError("unsupported_role")
        result.append({"role": role, "content": normalize_text(item.get("content"))})
    if len(result) < 2 or result[-1]["role"] != "assistant":
        raise RecordError("incomplete_final_exchange")
    return result


def content_hash(messages) -> str:
    return stable_hash([{"role": m["role"], "content": m["content"]} for m in messages])


def response_ids(messages, native_ids=None) -> list:
    """Per-message target identity: the exact preceding history of each Assistant reply.

    Threaded sources reuse native message IDs across releases, so their Assistant replies
    carry both the native ID and the history hash (``native_ids`` aligned with messages).
    """
    out = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            out.append(None)
            continue
        prefix = "content-prefix:" + stable_hash(
            [{"role": m["role"], "content": m["content"]} for m in messages[:index + 1]])
        out.append(f"oasst-native:{native_ids[index]}:{prefix}" if native_ids else prefix)
    return out


def source_key(source) -> str:
    for field in SOURCE_KEY_FIELDS + ("family", "native_source"):
        if not isinstance(source.get(field), str) or not source[field]:
            raise RecordError("incomplete_source_" + field)
    return canonical_json({k: source[k] for k in SOURCE_KEY_FIELDS})


def turn_bin(turns: int) -> str:
    return "1" if turns == 1 else "2-3" if turns <= 3 else "4-6" if turns <= 6 else "7+"


def prompt_group(messages) -> str | None:
    """Substantive identical first prompts form one group across sources (not greetings)."""
    first = next((m["content"] for m in messages if m["role"] == "user"), "")
    words = re.findall(r"\w+", first.casefold(), re.UNICODE)
    if len(words) < 6 or len(first) < 40:
        return None
    return "prompt-text:" + stable_hash(" ".join(words))


def group_ids(source, lineage, messages) -> tuple[str, ...]:
    """Isolation groups: conversations sharing any group are one family.

    Families never straddle train and validation, and only one member of a family is
    selected. Groups name the source row, native thread/prompt IDs and the first prompt.
    """
    ds, family, groups = source["dataset"], source["family"], []
    if family == "openassistant":
        tree = lineage["tree_id"]
        groups += [f"tree:{ds}:{tree}", f"oasst-tree:{tree}"]
        for node in lineage["message_path"]:
            groups += [f"lineage:{ds}:{node}", f"oasst-message:{node}"]
    elif family == "prejudge_dialogue":
        pointer = source["file_pointer"]
        groups.append(f"source-file-row:{ds}:{source['revision']}:"
                      f"{pointer['relative_path']}:{pointer['row_index']}")
        native = lineage.get("native", {})
        for key in ("id", "prompt_id", "conversation_id"):
            if native.get(key) is not None:
                groups.append(f"native-{key}:{ds}:{native[key]}")
    else:
        groups.append(f"source-row:{ds}:{source['revision']}:{source['row_id']}")
        if family == "no_robots" and lineage.get("prompt_id") is not None:
            groups.append(f"prompt-id:{ds}:{lineage['prompt_id']}")
    first = prompt_group(messages)
    if first:
        groups.append(first)
    return tuple(sorted(set(groups)))


@dataclass(frozen=True)
class Record:
    """One whole conversation. Later steps add facts with ``replace``; none edits text.

    ``rendered_tokens`` is set by the context check, ``sample`` (wave, stratum, inclusion
    probability) by candidate sampling, ``answers`` (raw judge answers) by labelling.
    Scores and filter flags are derived from ``answers`` in ``questions``/``selection``.
    """
    record_id: str
    content_hash: str
    source: dict
    messages: tuple
    group_ids: tuple
    lineage: dict
    rendered_tokens: int | None = None
    sample: dict | None = None
    answers: dict | None = None

    @property
    def turns(self) -> int:
        return sum(m["role"] == "assistant" for m in self.messages)

    @property
    def has_system(self) -> bool:
        return self.messages[0]["role"] == "system"

    def user_keys(self) -> list[str]:
        return [text_key(m["content"]) for m in self.messages if m["role"] == "user"]

    def to_json(self) -> str:
        value = asdict(self)
        value["messages"], value["group_ids"] = list(self.messages), list(self.group_ids)
        return canonical_json(value)

    @classmethod
    def from_json(cls, line: str) -> "Record":
        value = json.loads(line)
        value["messages"] = tuple(value["messages"])
        value["group_ids"] = tuple(value["group_ids"])
        return cls(**value)


def make_record(source, messages, lineage=None) -> Record:
    """The single entry: normalise, then derive identity and groups."""
    lineage = dict(lineage or {})
    msgs = normalize_messages(messages)
    native = None
    if source.get("family") == "openassistant":
        path = lineage.get("message_path")
        if not isinstance(path, list) or len(path) != len(msgs):
            raise RecordError("threaded_record_without_message_path")
        native = path
    digest = content_hash(msgs)
    rid = stable_hash({"source": source_key(source), "content_hash": digest,
                       "response_ids": response_ids(msgs, native)})
    canonical_json(lineage)
    return Record(record_id=rid, content_hash=digest, source=dict(source), messages=tuple(msgs),
                  group_ids=group_ids(source, lineage, msgs), lineage=lineage)


def with_groups(record: Record, extra) -> Record:
    """Union groups of a dropped duplicate into its survivor (deduplication only)."""
    return replace(record, group_ids=tuple(sorted(set(record.group_ids) | set(extra))))
