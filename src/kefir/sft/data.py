"""Whole-conversation SFT data: token rows, the one stream order, and update batches.

A token row is ``{"conversation_id", "input_ids", "assistant_mask"}``: the Renderer C
``token_ids`` and ``target_mask`` of one complete conversation (no truncation, no
packing). The checks in ``read_rows`` and the batch geometry in ``batch_tensors`` are the
ones the Kefir 1 SFT runs used: every row is padded with EOD to the full 2048 context and
only shifted Assistant targets are labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import torch

from kefir.trainer.types import TokenBatch

from .renderer import EOD, MAX_POSITIONS, MSG_START, ChatRenderer

CONTEXT = MAX_POSITIONS
PAD_TOKEN_ID = EOD
IGNORE_INDEX = -100


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_rows(path, *, context=CONTEXT, expected_count=None):
    """Read and validate token rows; any violation rejects the whole file."""
    rows, seen = [], set()
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        identity = row['conversation_id']
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError('empty or duplicate conversation identity')
        seen.add(identity)
        ids, mask = row['input_ids'], row['assistant_mask']
        if not 2 <= len(ids) <= context or len(mask) != len(ids):
            raise ValueError('invalid whole-chat token/mask length; no truncation allowed')
        if ids[0] != MSG_START or ids[-1] != EOD or mask[-1] is not True:
            raise ValueError('whole chat must start with message control and end with supervised Assistant EOD')
        if any(type(x) is not int or not 0 <= x < 8000 for x in ids):
            raise ValueError('invalid token id')
        if (any(type(x) is not bool for x in mask) or mask[0] or not any(mask[1:])
                or any(keep and token > EOD for token, keep in zip(ids, mask))):
            raise ValueError('invalid assistant mask or no shifted targets')
        rows.append(row)
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError('wrong conversation count')
    return rows


def render_rows(conversations, output, *, renderer: ChatRenderer | None = None) -> dict:
    """Render ``{"conversation_id", "messages"}`` records into a new token-row file.

    The file appears only when every record rendered.
    """
    from kefir.staging import staged_output
    count = targets = positions = 0
    seen = set()
    with staged_output(output) as temporary, temporary.open("x") as stream:
        renderer = renderer or ChatRenderer.load()
        for record in conversations:
            identity = record.get("conversation_id")
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError("empty or duplicate conversation identity")
            seen.add(identity)
            rendered = renderer.render(record["messages"])
            ids, mask = rendered["token_ids"], rendered["target_mask"]
            stream.write(json.dumps(dict(conversation_id=identity, input_ids=ids,
                                         assistant_mask=mask), separators=(",", ":")) + "\n")
            count += 1
            positions += len(ids)
            targets += sum(mask[1:])
    return {"conversations": count, "rendered_tokens": positions,
            "assistant_target_tokens": targets}


def shifted_labels(row):
    return [token if keep else IGNORE_INDEX for token, keep in
            zip(row['input_ids'][1:], row['assistant_mask'][1:], strict=True)]


def target_count(row) -> int:
    return sum(row['assistant_mask'][1:])


def segment_order(count: int, seed: int, order=None) -> list[int]:
    """The frozen order if given, else ``random.Random(seed)`` over ``range(count)``."""
    if order is None:
        order = list(range(count))
        random.Random(seed).shuffle(order)
        return order
    order = list(order)
    if sorted(order) != list(range(count)):
        raise ValueError('frozen order must consume each row exactly once')
    return order


@dataclass(frozen=True)
class Stream:
    """All training rows, and the batches (row indexes) in the one consumption order.

    Each segment is batched on its own, then segments are concatenated: this is how the
    10k screen and its 10k->15k continuation were consumed (313 + 157 updates at b32).
    """
    rows: tuple[dict, ...]
    batches: tuple[tuple[int, ...], ...]

    @classmethod
    def build(cls, segments, *, batch_size: int, seed: int) -> "Stream":
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("positive global batch size required")
        rows, batches, seen = [], [], set()
        for segment_rows, order in segments:
            identities = {row['conversation_id'] for row in segment_rows}
            if identities & seen:
                raise ValueError('a conversation repeats across segments')
            seen |= identities
            offset = len(rows)
            ordered = segment_order(len(segment_rows), seed, order)
            batches.extend(tuple(offset + i for i in ordered[start:start + batch_size])
                           for start in range(0, len(ordered), batch_size))
            rows.extend(segment_rows)
        if not batches:
            raise ValueError("empty training stream")
        return cls(tuple(rows), tuple(batches))

    def chats_after(self, steps: int) -> int:
        return sum(len(batch) for batch in self.batches[:steps])

    def targets_after(self, steps: int) -> int:
        return sum(target_count(self.rows[i]) for batch in self.batches[:steps] for i in batch)

    def targets_at_chats(self, chats: int) -> int:
        """Assistant targets in the first ``chats`` conversations of the stream order."""
        if not 0 <= chats <= len(self.rows):
            raise ValueError("chat boundary outside the stream")
        order = [i for batch in self.batches for i in batch]
        return sum(target_count(self.rows[i]) for i in order[:chats])

    def prefix_sha256(self, steps: int) -> str:
        """Identity of the consumed prefix: conversation ids, batch by batch."""
        return canonical_sha256([[self.rows[i]['conversation_id'] for i in batch]
                                 for batch in self.batches[:steps]])


def batch_tensors(rows, step: int, *, context=CONTEXT) -> TokenBatch:
    """Pad every row to the full context with EOD; labels are shifted Assistant targets."""
    inputs = torch.full((len(rows), context), PAD_TOKEN_ID, dtype=torch.long)
    labels = torch.full_like(inputs, IGNORE_INDEX)
    for i, row in enumerate(rows):
        n = len(row['input_ids']) - 1
        inputs[i, :n] = torch.tensor(row['input_ids'][:-1])
        labels[i, :n] = torch.tensor(shifted_labels(row))
    targets = int((labels != IGNORE_INDEX).sum())
    if targets <= 0:
        raise ValueError('zero-target update')
    return TokenBatch(block_id=step, split='train', input_ids=inputs, labels=labels,
                      sequence_count=len(rows), target_token_count=targets)


__all__ = ["CONTEXT", "PAD_TOKEN_ID", "Stream", "batch_tensors", "canonical_sha256",
           "read_rows", "render_rows", "segment_order", "shifted_labels", "target_count"]
