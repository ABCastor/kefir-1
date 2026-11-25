"""Renderer C shared by supervised examples and conversational inference.

Implementation qualification is separate from owner adoption of a chat recipe.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from kefir.tokenizer import TOKENIZER_SHA256

MSG_START, MSG_END, EOD, MAX_POSITIONS = 7993, 7994, 7992, 2048


class ContextOverflow(ValueError):
    pass


class ChatRenderer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.identity = {
            "tokenizer_sha256": TOKENIZER_SHA256, "eod_token_id": EOD,
            "renderer_revision": "kefir-chat-c-v1",
            "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "qualification": "ops_preparation", "scientifically_qualified": False,
        }

    @classmethod
    def load(cls):
        from kefir.tokenizer import load_tokenizer
        return cls(load_tokenizer())

    def encode_content(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("nonempty chat content required")
        ids = self.tokenizer.encode(text)
        if not ids or any(type(x) is not int or not 0 <= x < EOD for x in ids):
            raise ValueError("content must encode as ordinary text IDs")
        return ids

    def _render(self, messages, *, prompt=False):
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages required")
        ids, mask, spans, expected = [], [], [], "user"
        for index, message in enumerate(messages):
            role = message.get("role")
            if role == "system" and index == 0:
                pass
            elif role != expected:
                raise ValueError("alternating complete exchanges required")
            else:
                expected = "assistant" if role == "user" else "user"
            header = [MSG_START] + self.encode_content(role + "\n")
            body = self.encode_content(message["content"])
            start = len(ids) + len(header)
            ids.extend(header + body + [EOD if role == "assistant" else MSG_END])
            mask.extend([False] * len(header) + [role == "assistant"] * (len(body) + 1))
            if role == "assistant":
                spans.append({"message_index": index, "start": start,
                              "end": start + len(body), "eod_index": start + len(body)})
        if prompt:
            if messages[-1]["role"] != "user":
                raise ValueError("generation requires a final User message")
            ids.extend([MSG_START] + self.encode_content("assistant\n"))
        elif messages[-1]["role"] != "assistant" or not spans:
            raise ValueError("supervised chat requires a final Assistant reply")
        if len(ids) > MAX_POSITIONS:
            raise ContextOverflow("whole_conversation_exceeds_2048")
        return {"token_ids": ids, "target_mask": mask, "response_spans": spans}

    def render(self, messages):
        return self._render(messages)

    def generation_prompt(self, messages):
        return self._render(messages, prompt=True)["token_ids"]
