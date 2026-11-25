"""Independent metrics, no composite score. Case constraints require ASCII letters."""
import math
import re
from collections import defaultdict

_TERMINATIONS = {"native_end", "output_cap", "context_exhausted", "error"}
_ROLE = re.compile(r"(?:^|\n)\s*(?:user|assistant|system|developer)\s*:|<\|(?:user|assistant|system|developer)(?:\|>|>)|<\|im_start\|>\s*(?:user|assistant|system|developer)\b|\[/?INST\]", re.I)


def _id(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("id must be a nonempty, trimmed string")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _logp(value, name):
    value = _number(value, name)
    if value > 0:
        raise ValueError(f"{name} must be non-positive")
    return value


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _word(text, word):
    return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, re.I) is not None


def _constraint(spec, text):
    if not isinstance(spec, dict):
        raise ValueError("constraint must be an object")
    kind = spec.get("type")
    if kind in {"raw_exact", "forbidden_word", "required_word", "ends_with"}:
        value = spec.get("value")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{kind} requires a nonempty value")
        if kind == "raw_exact":
            return text == value
        if kind == "ends_with":
            return text.endswith(value)
        found = _word(text, value)
        return found if kind == "required_word" else not found
    if kind == "lowercase":
        return bool(re.search(r"[A-Za-z]", text)) and text == text.lower()
    if kind == "uppercase":
        return bool(re.search(r"[A-Za-z]", text)) and text == text.upper()
    if kind in {"bullet_lines", "line_count"}:
        count = _positive_int(spec.get("count"), "count")
        lines = text.splitlines()
        if kind == "line_count":
            return len(lines) == count
        prefix = spec.get("prefix", "- ")
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("bullet prefix must be nonempty")
        return len(lines) == count and all(line.startswith(prefix) and line[len(prefix):].strip() for line in lines)
    raise ValueError(f"unknown constraint: {kind!r}")


def score_response(case, response_record):
    """Score one case, raising ValueError for malformed or incomplete observations."""
    if not isinstance(case, dict) or not isinstance(response_record, dict):
        raise ValueError("case and response must be objects")
    case_id = _id(case.get("id"))
    if _id(response_record.get("id")) != case_id:
        raise ValueError("case and response IDs differ")
    mode = case.get("content_mode")
    if mode not in {"exact", "review", "grammar"}:
        raise ValueError("unknown content mode")
    row = {"id": case_id, "content_correct": None, "strict_exact": None,
           "constraints": [], "constraints_pass": None, "joint_success": None,
           "native_termination": None, "role_leakage": None, "grammar_credit": None,
           "grammar_margin": None, "choice_credit": None, "choice_tie_count": None, "choice_raw_credit": None, "choice_raw_tie_count": None,
           "gold_bpb": None}
    if mode == "grammar":
        good = _logp(response_record.get("logp_good"), "logp_good")
        bad = _logp(response_record.get("logp_bad"), "logp_bad")
        row["grammar_margin"] = _number(good - bad, "grammar margin")
        row["grammar_credit"] = 1.0 if good > bad else 0.5 if good == bad else 0.0
    else:
        text = response_record.get("text")
        if not isinstance(text, str):
            raise ValueError("generated response text must be a string")
        termination = response_record.get("termination")
        if termination not in _TERMINATIONS:
            raise ValueError("unexpected termination")
        gold = case.get("gold")
        if not isinstance(gold, list) or not gold or any(not isinstance(x, str) or not x.strip() for x in gold):
            raise ValueError("gold must be a nonempty list of nonempty strings")
        nonempty = bool(text.strip())
        row["strict_exact"] = termination != "error" and nonempty and text.strip().casefold() in {x.strip().casefold() for x in gold}
        if mode == "exact":
            row["content_correct"] = row["strict_exact"]
        else:
            adjudication = response_record.get("adjudication")
            if adjudication is not None and not isinstance(adjudication, bool):
                raise ValueError("adjudication must be a boolean or null")
            row["content_correct"] = (adjudication and nonempty) if adjudication is not None else None
        if termination == "error":
            row["content_correct"] = False
        constraints = case.get("constraints", [])
        if not isinstance(constraints, list):
            raise ValueError("constraints must be a list")
        row["constraints"] = [{"type": c.get("type") if isinstance(c, dict) else None,
                               "pass": bool(_constraint(c, text) and nonempty)} for c in constraints]
        row["constraints_pass"] = nonempty and all(c["pass"] for c in row["constraints"])
        content = row["content_correct"]
        row["joint_success"] = False if not row["constraints_pass"] or content is False else (True if content is True else None)
        row["native_termination"] = termination == "native_end"
        row["role_leakage"] = bool(_ROLE.search(text))
        row["termination"] = termination
        row["empty"] = not nonempty
    if "choice_logps" in response_record or "choice_char_lengths" in response_record:
        choices = case.get("choices")
        logps = response_record.get("choice_logps")
        lengths = response_record.get("choice_char_lengths")
        if not isinstance(choices, list) or len(choices) < 2 or any(not isinstance(c, str) or not c for c in choices):
            raise ValueError("choices must contain at least two nonempty strings")
        if not isinstance(logps, list) or not isinstance(lengths, list) or len(logps) != len(choices) or len(lengths) != len(choices):
            raise ValueError("choice likelihoods and lengths must cover all choices")
        raw = [_logp(v, "choice logp") for v in logps]
        vals = [v / _positive_int(n, "choice length") for v, n in zip(raw, lengths)]
        if lengths != [len(choice) for choice in choices]:
            raise ValueError("choice character lengths do not match choices")
        answer = case.get("gold_choice")
        if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(choices):
            raise ValueError("invalid gold choice")
        winners = [i for i, value in enumerate(vals) if value == max(vals)]
        raw_winners = [i for i, value in enumerate(raw) if value == max(raw)]
        row["choice_raw_credit"] = 1.0 / len(raw_winners) if answer in raw_winners else 0.0
        row["choice_raw_tie_count"] = len(raw_winners)
        row["choice_credit"] = 1.0 / len(winners) if answer in winners else 0.0
        row["choice_tie_count"] = len(winners)
        row["choice_normalization"] = "log_probability_per_unicode_character"
        row["choice_normalized_logps"] = vals
    if "sum_logprob" in response_record or "gold_bytes" in response_record:
        logp = _logp(response_record.get("sum_logprob"), "sum_logprob")
        size = _positive_int(response_record.get("gold_bytes"), "gold_bytes")
        gold = case.get("gold")
        if not isinstance(gold, list) or not gold or not isinstance(gold[0], str) or size != len(gold[0].encode("utf-8")):
            raise ValueError("gold byte count does not match canonical answer")
        row["gold_bpb"] = -logp / (size * math.log(2))
    return row


def _stats(values):
    known = [v for v in values if v is not None]
    return {"n": len(values), "scored": len(known), "mean": sum(known) / len(known) if known else None}


def aggregate(cases, scored_rows):
    """Report category and pair statistics; reject missing, extra, or duplicate IDs."""
    if not isinstance(cases, (list, tuple)) or not isinstance(scored_rows, (list, tuple)):
        raise ValueError("cases and scored rows must be sequences")
    def index(records):
        out = {}
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("records must be objects")
            identifier = _id(record.get("id"))
            if identifier in out:
                raise ValueError("duplicate ID: " + identifier)
            out[identifier] = record
        return out
    by_id, results = index(cases), index(scored_rows)
    if by_id.keys() != results.keys():
        raise ValueError(f"ID mismatch: missing={sorted(by_id.keys() - results.keys())}, extra={sorted(results.keys() - by_id.keys())}")
    slices, pairs = defaultdict(list), defaultdict(list)
    for identifier, case in by_id.items():
        row = results[identifier]
        category = case.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError("category must be a nonempty string")
        slices[category].append(row)
        if "pair_id" in case:
            pair = _id(case["pair_id"])
            variant = _id(case.get("variant"))
            pairs[pair].append((case, row, variant))
    metrics = ("strict_exact", "content_correct", "constraints_pass", "joint_success", "native_termination", "role_leakage", "grammar_credit", "grammar_margin", "choice_credit", "choice_raw_credit", "gold_bpb")
    report = {"categories": {category: {metric: _stats([row.get(metric) for row in rows]) for metric in metrics} for category, rows in sorted(slices.items())},
              "pairs": {}}
    grouped = defaultdict(list)
    for pair_id, members in pairs.items():
        if len(members) != 2 or len({member[2] for member in members}) != 2:
            raise ValueError(f"pair {pair_id} must have exactly two distinct variants")
        if len({member[0].get("family_id") for member in members}) != 1:
            raise ValueError(f"pair {pair_id} has inconsistent families")
        vals = [member[1].get("content_correct") for member in members]
        both = False if False in vals else True if vals == [True, True] else None
        grouped[members[0][0]["category"]].append(both)
    for category, rows in slices.items():
        generated = [row for row in rows if row.get("termination") is not None]
        constraints = [item["pass"] for row in rows for item in row.get("constraints", [])]
        report["categories"][category]["instruction_level"] = _stats(constraints)
        report["categories"][category]["grammar_ties"] = sum(row.get("grammar_credit") == 0.5 for row in rows)
        report["categories"][category]["choice_ties"] = sum((row.get("choice_tie_count") or 0) > 1 for row in rows)
        report["categories"][category]["termination_counts"] = {t: sum(row.get("termination") == t for row in generated) for t in sorted(_TERMINATIONS)}
        report["categories"][category]["runtime_error_count"] = sum(row.get("termination") == "error" for row in generated)
        report["categories"][category]["empty_count"] = sum(row.get("empty") is True for row in generated)
    report["pairs"] = {category: {"both_correct": _stats(values)} for category, values in sorted(grouped.items())}
    # Keep source and task differences visible rather than hiding a floor in the mean.
    breakdowns = defaultdict(list)
    for identifier, c in by_id.items():
        source=c.get('source',{})
        labels={'source':source.get('dataset',source.get('kind')),
                'popularity':source.get('popularity_stratum'),
                'subtask':c.get('subtask'),'difficulty':c.get('difficulty'),
                'context_type':c.get('context_type'),'example_type':c.get('example_type'),
                'instruction_family':c.get('instruction_family'),'stopping_type':c.get('stopping_type')}
        for dimension,value in labels.items():
            if value is not None:
                breakdowns[f'{c["category"]}/{dimension}/{value}'].append(results[identifier])
    report['breakdowns']={key:{metric:_stats([row.get(metric) for row in rows]) for metric in metrics}
                          for key,rows in sorted(breakdowns.items())}
    return report
