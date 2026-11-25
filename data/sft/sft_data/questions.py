"""The ten judge questions, the answer format and the answer-to-score mapping.

``questions.json`` holds the frozen question set exactly as it labelled the published pool
(its SHA-256 below is the pool's ``schema_hash``) and the short evaluator preamble placed
before every conversation. The judge sees the conversation as plain text (``state``) and
answers each question separately.

Answer types:
- ``noul``: a yes/no question; the answer is the probability of YES, ``{"type": "noul",
  "noul": p}`` with 0 <= p <= 1. The score is ``p``.
- ``score``: an ordinal rubric with five levels 0..4; ``{"type": "score", "score": s,
  "probabilities": {"0": .., "4": ..}}`` with ``s`` the expected level. The score is s / 4.
- ``choice``: one of named options; ``{"type": "choice", "choice": k, "probabilities":
  {...}}``. Categorical, not scored.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SCHEMA_SHA256 = "3a0c8f02c26d490b1701df3c0fd71d988991eab6aa0ce62a5f8136eac418ac8f"
NUMERIC = ("english_only", "excluded_formalism", "english_quality", "instruction_fidelity",
           "content_error", "reasoning_required", "context_required", "factual_uncertainty")
CATEGORICAL = ("task", "subject")


class AnswerError(ValueError):
    """An answer that does not match its question; the conversation stays unlabelled."""


def schema_hash(questions) -> str:
    return hashlib.sha256(json.dumps(questions, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _load():
    data = json.loads(Path(__file__).with_name("questions.json").read_text(encoding="utf-8"))
    if schema_hash(data["questions"]) != SCHEMA_SHA256:
        raise RuntimeError("questions.json differs from the frozen question set")
    return data["state_prefix"], data["questions"]


STATE_PREFIX, QUESTIONS = _load()


def options(question) -> list[str]:
    """Answer keys a judge must assign probabilities to."""
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["yes", "no"]


def state(messages) -> str:
    """The text the judge reads: preamble, then every message with its index and role."""
    body = "\n\n".join(f"MESSAGE {i} ROLE {m['role']}\n{m['content']}\nEND MESSAGE {i}"
                       for i, m in enumerate(messages))
    return STATE_PREFIX + "CONVERSATION_START\n" + body + "\nCONVERSATION_END"


# JSON Schema of one answer object, for judges that support structured output.
ANSWER_SCHEMA = {
    "oneOf": [
        {"type": "object", "required": ["type", "noul"],
         "properties": {"type": {"const": "noul"}, "noul": {"type": "number", "minimum": 0, "maximum": 1}}},
        {"type": "object", "required": ["type", "score"],
         "properties": {"type": {"const": "score"}, "score": {"type": "number", "minimum": 0, "maximum": 4},
                        "probabilities": {"type": "object", "additionalProperties": {"type": "number"}}}},
        {"type": "object", "required": ["type", "choice"],
         "properties": {"type": {"const": "choice"}, "choice": {"type": "string"},
                        "probabilities": {"type": "object", "additionalProperties": {"type": "number"}}}},
    ]
}


def _unit(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AnswerError("non-numeric answer: " + name)
    return float(value)


def parse_answers(answers, questions=QUESTIONS) -> dict:
    """Validate raw answers against the questions; return them unchanged."""
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise AnswerError("answers must cover exactly the ten questions")
    for name, question in questions.items():
        answer = answers[name]
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise AnswerError("wrong answer type: " + name)
        probabilities = answer.get("probabilities")
        if probabilities is not None:
            if not isinstance(probabilities, dict) or set(probabilities) != set(options(question)):
                raise AnswerError("probabilities must name every option: " + name)
            if any(not 0 <= _unit(v, name) <= 1 for v in probabilities.values()):
                raise AnswerError("probability outside 0..1: " + name)
        if question["type"] == "noul":
            if not 0 <= _unit(answer.get("noul"), name) <= 1:
                raise AnswerError("YES probability outside 0..1: " + name)
        elif question["type"] == "score":
            if not 0 <= _unit(answer.get("score"), name) <= len(question["criteria"]) - 1:
                raise AnswerError("ordinal score outside the scale: " + name)
        elif answer.get("choice") not in question["criteria"]:
            raise AnswerError("unknown choice: " + name)
    return answers


def scores(answers, questions=QUESTIONS) -> dict:
    """The eight numeric scores in 0..1: YES probability, or ordinal level / 4."""
    parse_answers(answers, questions)
    out = {}
    for name, question in questions.items():
        if question["type"] == "noul":
            out[name] = answers[name]["noul"]
        elif question["type"] == "score":
            out[name] = answers[name]["score"] / (len(question["criteria"]) - 1)
    return out


def categories(answers, questions=QUESTIONS) -> dict:
    parse_answers(answers, questions)
    return {name: answers[name]["choice"] for name, q in questions.items() if q["type"] == "choice"}
