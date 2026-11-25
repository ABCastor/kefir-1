"""Conservative pre-filters applied before labelling.

Three checks; the first that fires excludes the conversation:

1. Source domain, applied by ``prepare`` before deduplication: a native category that
   names an excluded domain (No Robots ``Coding``).
2. Scope: presence tests over every message of every role for non-Latin letters, formal
   logic or set symbols, LaTeX, markup, SQL, JSON objects, program code, shell commands and
   symbolic variable relations. Presence, not density: one fragment excludes. Ordinary
   Markdown and numeric arithmetic are not matched.
3. Language, only on scope survivors: each message and each paragraph with at least 12
   words is checked with fastText ``lid.176.bin``; a non-English fastText label with score
   >= 0.8 is rechecked with Lingua. The conversation is excluded only when both agree on
   the same non-English language with fastText >= 0.95, Lingua >= 0.98 and a Lingua margin
   >= 0.8 over its second language.

Passing does not certify English or the absence of code; the judge questions do that.
The language detectors are injected so that this module imports nothing outside the
standard library (see ``language_detectors``).
"""
from __future__ import annotations

import ast
import functools
import json
import math
import re
import unicodedata

SCOPE_VERSION = "scope-prefilter-v2.1"
LANGUAGE_VERSION = "language-scope-v1"
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_SHA256 = "7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e"
LANGUAGE = {"minimum_words": 12, "fasttext_for_lingua": .8, "fasttext_joint": .95,
            "lingua_joint": .98, "lingua_margin": .8}
EXCLUDED_NATIVE_CATEGORIES = {"HuggingFaceH4/no_robots": {"coding"}}

FORMAL = set("∀∃∄∈∉⊂⊃⊆⊇∪∩∧∨¬⊢⊨∑∏∫∂∇≈≡")
PATTERNS = {
    "latex": re.compile(r"\\(?:begin|end|frac|dfrac|sqrt|mathbb|mathbf|mathrm|text|forall|exists|sum|prod|int|partial|nabla|alpha|beta|gamma|theta|lambda|rightarrow|implies)\b(?:\{|\s|_|\^|$)|\\[\[\]()]"),
    "xml": re.compile(r"<([A-Za-z][\w:.-]*)\b[^<>]*>[^<>]*</\1\s*>|<[A-Za-z][\w:.-]*(?:\s+[\w:.-]+\s*=\s*[\"'][^\"']*[\"'])+\s*/>|<\?xml\s"),
    "sql": re.compile(r"\bSELECT\s+(?:\*|[\w.,\s]+)\s+FROM\s+\w+\s*(?:;|WHERE\b|$)|\bCREATE\s+TABLE\s+\w+\s*\(", re.M),
    "programming_syntax": re.compile(r"(?m)^\s*(?:def\s+\w+\s*\([^\n]*\)\s*:|(?:async\s+)?function\s+\w*\s*\([^\n]*\)\s*\{|(?:const|let|var)\s+\w+\s*=|(?:public\s+)?(?:static\s+)?void\s+\w+\s*\(|#include\s*[<\"]|console\.log\s*\(|System\.out\.println\s*\(|(?:pip|npm)\s+install\s+|#!/(?:usr/)?bin/)"),
    "python_control": re.compile(r"(?m)^\s*(?:for\s+\w+\s+in\s+[^\n:]+:|(?:if|elif|while)\s+[^\n:]+:\s*(?:(?:pass|continue|break)\s*(?:#.*)?$|\w+\s*\())"),
    "shell_command": re.compile(r"(?m)^\s*(?:echo\s+[\"'$]|(?:ls|cat|grep|curl|wget|chmod|sudo)\s+-[A-Za-z]|(?:cd|mkdir)\s+[/~.]|export\s+[A-Z_][A-Z0-9_]*=)"),
    "variable_relation": re.compile(r"\b[A-Za-z]\s*(?:≤|≥|≠|:=)\s*(?:[A-Za-z]\b|[0-9])"),
}
CALL = re.compile(r"\b(?:print|range|len|json\.loads|json\.dumps)\s*\([^\n)]*\)")


def source_domain(record) -> str | None:
    excluded = EXCLUDED_NATIVE_CATEGORIES.get(record.source["dataset"], set())
    category = record.lineage.get("native_category")
    if isinstance(category, str) and category.strip().casefold() in excluded:
        return "source_domain"
    return None


def _json_object(text) -> bool:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except (ValueError, RecursionError):
            continue
        if (isinstance(value, dict) and value) or (
                isinstance(value, list) and any(isinstance(v, dict) and v for v in value)):
            return True
    return False


def _python_import(text) -> bool:
    for line in text.splitlines():
        if not re.match(r"^\s*(?:from\s+\w+[.\w]*\s+import\s+|import\s+[\w.]+)", line):
            continue
        try:
            body = ast.parse(line.strip()).body
        except (SyntaxError, ValueError):
            continue
        if body and all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in body):
            return True
    return False


def _executable_call(text) -> bool:
    for call in CALL.finditer(text):
        try:
            expr = ast.parse(call.group(), mode="eval").body
        except (SyntaxError, ValueError):
            continue
        isolated = any(line.strip().rstrip(";") == call.group() for line in text.splitlines())
        literal = isinstance(expr, ast.Call) and any(
            isinstance(a, (ast.Constant, ast.List, ast.Dict, ast.Tuple)) for a in expr.args)
        if isolated or literal:
            return True
    return False


def scope_reasons(messages) -> list[str]:
    reasons = set()
    for message in messages:
        text = message["content"]
        if any(not c.isascii() and unicodedata.category(c).startswith("L")
               and "LATIN" not in unicodedata.name(c, "") and "MODIFIER LETTER" not in unicodedata.name(c, "")
               for c in text):
            reasons.add("non_latin_letter")
        if any(c in FORMAL for c in text):
            reasons.add("formal_notation")
        reasons.update(rule for rule, pattern in PATTERNS.items() if pattern.search(text))
        if _json_object(text):
            reasons.add("json_object")
        if _python_import(text):
            reasons.add("python_import")
        if _executable_call(text):
            reasons.add("programming_syntax")
    return sorted(reasons)


def segments(text):
    return list(dict.fromkeys([text] + re.split(r"\n\s*\n", text)))


def language_check(fasttext_predict, lingua_scores):
    """Build the per-segment check from two callables.

    ``fasttext_predict(text) -> (label, score)`` (top label without ``__label__``);
    ``lingua_scores(text) -> {iso_639_1_code: confidence}``.
    """
    @functools.lru_cache(maxsize=4096)
    def check(text) -> bool:
        words = len(re.findall(r"[^\W\d_]+", text))
        if not text.strip() or words < LANGUAGE["minimum_words"]:
            return False
        language, score = fasttext_predict(text.replace("\n", " "))
        if not math.isfinite(score) or not 0 <= score <= 1.0001:
            raise ValueError("invalid fastText score")
        if language == "en" or score < LANGUAGE["fasttext_for_lingua"]:
            return False
        ranked = sorted(lingua_scores(text).items(), key=lambda x: x[1], reverse=True)
        if len(ranked) < 2:
            raise ValueError("Lingua must score at least two languages")
        (lingua_language, top), (_, second) = ranked[0], ranked[1]
        return (lingua_language == language and score >= LANGUAGE["fasttext_joint"]
                and top >= LANGUAGE["lingua_joint"] and top - second >= LANGUAGE["lingua_margin"])
    return check


def language_detectors(fasttext_model_path):
    """fastText lid.176 plus Lingua over all its languages (optional dependencies)."""
    import fasttext
    from lingua import LanguageDetectorBuilder
    model = fasttext.load_model(str(fasttext_model_path))
    detector = LanguageDetectorBuilder.from_all_languages().build()

    def predict(text):
        labels, probabilities = model.predict(text, k=2)
        return labels[0].replace("__label__", ""), float(probabilities[0])

    def scores(text):
        return {v.language.iso_code_639_1.name.lower(): v.value
                for v in detector.compute_language_confidence_values(text)}
    return language_check(predict, scores)


def exclusion(record, check=None) -> str | None:
    """Scope, then language: the first exclusion reason, or None when the conversation passes.

    The source-domain rule runs earlier, in ``prepare``, before deduplication.
    """
    reasons = scope_reasons(record.messages)
    if reasons:
        return "scope:" + ",".join(reasons)
    if check is not None and any(check(s) for m in record.messages for s in segments(m["content"])):
        return "language"
    return None
