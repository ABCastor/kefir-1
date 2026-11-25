"""CPU tests on tiny synthetic conversations: python -m pytest data/sft -q (from the repository root)"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json

import pytest

from sft_data import judge, prefilter, questions, sample, sources
from sft_data import selection as select
from sft_data.dedup import deduplicate
from sft_data.records import Record, RecordError, make_record, normalize_messages, text_key

LONG = "Please explain in a few plain sentences how the water cycle moves water around the planet"


def source(row, dataset="HuggingFaceTB/smol-smoltalk", split="train"):
    spec = sources.SOURCES[dataset]
    return {"dataset": dataset, "revision": spec.revision, "config": spec.config, "split": split,
            "row_id": f"{split}:data/x.parquet:{row}", "family": spec.family, "native_source": "everyday"}


def chat(row, user=LONG, reply="Water evaporates, condenses into clouds and falls as rain.", **kw):
    return make_record(source(row, **kw), [{"role": "user", "content": user}, {"role": "assistant", "content": reply}],
                       {"native_category": "everyday"})


def answers(**over):
    base = {"english_only": .995, "excluded_formalism": .001, "english_quality": 3.8,
            "instruction_fidelity": .97, "content_error": .01, "reasoning_required": .1,
            "context_required": .1, "factual_uncertainty": .1}
    base.update(over)
    out = {}
    for name, q in questions.QUESTIONS.items():
        if q["type"] == "noul":
            out[name] = {"type": "noul", "noul": base[name]}
        elif q["type"] == "score":
            out[name] = {"type": "score", "score": base[name]}
        else:
            out[name] = {"type": "choice", "choice": over.get(name, next(iter(q["criteria"])))}
    return out


def labelled(record, wave="original", tokens="short", **over):
    stratum = json.dumps([record.source["dataset"], "1", tokens, False], separators=(",", ":"))
    return replace(record, answers=answers(**over), sample={"wave": wave, "stratum": stratum,
                                                             "inclusion_probability": .5})


# Records ---------------------------------------------------------------------------------

def test_normalisation_and_identity_are_stable():
    a = chat(1, user=LONG + "\r\n")
    b = chat(1, user=LONG + "\n")
    assert a.messages[0]["content"].endswith("\n") and a.record_id == b.record_id
    assert chat(2).record_id != a.record_id  # the source row is part of the identity
    assert Record.from_json(a.to_json()) == a


def test_whole_conversations_only():
    with pytest.raises(RecordError):
        normalize_messages([{"role": "user", "content": "Hi"}])
    with pytest.raises(RecordError):
        normalize_messages([{"role": "user", "content": "Hi"}, {"role": "system", "content": "x"},
                            {"role": "assistant", "content": "Hello"}])


def test_flat_and_threaded_sources():
    dolly = sources.dialogue_record({"instruction": "Summarise this.", "context": "A long text.", "response": "Short."},
                                    "databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", 3, "train")
    assert dolly.messages[0]["content"] == "Summarise this.\n\nContext:\nA long text."
    sys_chat = sources.dialogue_record(
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
         "chat_template_kwargs": {"custom_instructions": "Be a knight.", "enable_thinking": False}},
        "HuggingFaceTB/smoltalk2", "f.parquet", 0, "train")
    assert sys_chat.has_system and sys_chat.messages[0]["content"] == "Be a knight."
    tree = [  # root prompt, two replies, one follow-up below the first reply, one deleted reply
        {"message_id": "r", "parent_id": None, "message_tree_id": "t", "role": "prompter", "lang": "en", "text": "Q?"},
        {"message_id": "a", "parent_id": "r", "message_tree_id": "t", "role": "assistant", "lang": "en", "text": "A."},
        {"message_id": "b", "parent_id": "r", "message_tree_id": "t", "role": "assistant", "lang": "en", "text": "B."},
        {"message_id": "f", "parent_id": "a", "message_tree_id": "t", "role": "prompter", "lang": "en", "text": "More?"},
        {"message_id": "g", "parent_id": "f", "message_tree_id": "t", "role": "assistant", "lang": "en", "text": "Yes."},
        {"message_id": "d", "parent_id": "r", "message_tree_id": "t", "role": "assistant", "lang": "en",
         "text": "D.", "deleted": True},
    ]
    records, rejected = sources.thread_records(tree, "OpenAssistant/oasst1", "train",
                                               [{"relative_path": "f", "row_index": i} for i in range(len(tree))])
    assert sorted(r.lineage["message_path"] for r in records) == [["r", "a", "f", "g"], ["r", "b"]]
    assert rejected == 1 and all("oasst-tree:t" in r.group_ids for r in records)
    short = sources.ancestor(next(r for r in records if r.turns == 2), 2)
    assert short.turns == 1 and "oasst-message:g" in short.group_ids


# Pre-filters and deduplication -----------------------------------------------------------

def test_scope_rules_are_presence_tests():
    plain = chat(1)
    assert prefilter.exclusion(plain) is None
    code = chat(2, reply="Use this:\ndef f(x):\n    return x")
    latex = chat(3, reply=r"The area is \frac{1}{2} b h.")
    assert prefilter.exclusion(code).startswith("scope:") and "latex" in prefilter.exclusion(latex)
    assert prefilter.exclusion(chat(4, reply="Two plus three is 5, and 10 / 2 = 5.")) is None


def test_language_gate_needs_both_detectors():
    foreign = chat(1, reply="Questa è una risposta abbastanza lunga scritta interamente in italiano per la prova.")
    agree = prefilter.language_check(lambda t: ("it", .99), lambda t: {"it": .99, "es": .01})
    disagree = prefilter.language_check(lambda t: ("it", .99), lambda t: {"es": .99, "it": .01})
    unsure = prefilter.language_check(lambda t: ("it", .9), lambda t: {"it": .99, "es": .01})
    assert prefilter.exclusion(foreign, agree) == "language"
    assert prefilter.exclusion(foreign, disagree) is None and prefilter.exclusion(foreign, unsure) is None


def test_source_domain_rule():
    spec = sources.SOURCES["HuggingFaceH4/no_robots"]
    row = {"messages": [{"role": "user", "content": "Write code"}, {"role": "assistant", "content": "Sure."}],
           "category": "Coding", "prompt_id": "p"}
    record = sources.dialogue_record(row, "HuggingFaceH4/no_robots", spec.train[0], 0, "train")
    assert prefilter.source_domain(record) == "source_domain"


def test_deduplication_and_heldout_isolation():
    a = chat(1)
    duplicate = chat(2)  # same text, other row: one survivor inherits both groups
    heldout = chat(3, user=LONG + " please", split="test")
    linked = chat(4, user=LONG + " please", reply="Different reply.")  # same first prompt as heldout
    out, counts = deduplicate([a, duplicate, heldout, linked])
    assert [r.record_id for r in out] == [min(a.record_id, duplicate.record_id)]
    assert set(a.group_ids) | set(duplicate.group_ids) <= set(out[0].group_ids)
    assert counts["exact_duplicate_removed"] == 1 and counts["heldout_family_removed"] == 1


# Questions, scores and the judge ---------------------------------------------------------

def test_question_set_is_the_frozen_one():
    assert questions.schema_hash(questions.QUESTIONS) == questions.SCHEMA_SHA256
    assert list(questions.QUESTIONS) == ["english_only", "excluded_formalism", "english_quality",
                                         "instruction_fidelity", "content_error", "task", "subject",
                                         "reasoning_required", "context_required", "factual_uncertainty"]
    text = questions.state([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}])
    assert text.endswith("CONVERSATION_START\nMESSAGE 0 ROLE user\nHi\nEND MESSAGE 0\n\n"
                         "MESSAGE 1 ROLE assistant\nHello\nEND MESSAGE 1\nCONVERSATION_END")


def test_score_mapping():
    s = questions.scores(answers(english_quality=3.0, english_only=.42))
    assert s["english_quality"] == .75 and s["english_only"] == .42 and len(s) == 8
    assert questions.categories(answers(subject="arts"))["subject"] == "arts"


@pytest.mark.parametrize("bad", [
    {"english_only": {"type": "noul", "noul": 1.2}},
    {"english_quality": {"type": "score", "score": 5}},
    {"task": {"type": "choice", "choice": "coding"}},
    {"english_only": {"type": "score", "score": 1}},
])
def test_invalid_answers_are_rejected(bad):
    with pytest.raises(questions.AnswerError):
        questions.parse_answers({**answers(), **bad})


def test_chat_judge_turns_stated_probabilities_into_answers():
    def post(url, payload, headers, timeout):
        prompt = payload["messages"][1]["content"]
        q = questions.QUESTIONS[prompt.split("QUESTION ", 1)[1].split("\n", 1)[0]]
        if q["type"] == "noul":
            body = {"yes": .2, "no": .8}
        elif q["type"] == "score":
            body = {"0": 0, "1": 0, "2": 0, "3": .5, "4": .5}
        else:
            keys = questions.options(q)
            body = {k: (1 if k == keys[1] else 0) for k in keys}
        return {"choices": [{"message": {"content": "Answer: " + json.dumps(body)}}]}
    out = judge.ChatJudge("http://localhost:1/v1", "m", post=post).answer("STATE")
    assert questions.scores(out)["english_quality"] == pytest.approx(3.5 / 4)
    assert questions.scores(out)["content_error"] == pytest.approx(.2)
    assert questions.categories(out)["task"] == "knowledge"


# Selection -------------------------------------------------------------------------------

@pytest.mark.parametrize("over,strict,intermediate", [
    ({}, True, True),
    ({"english_only": .99}, True, True),
    ({"english_only": .989}, False, True),
    ({"english_quality": 3.6}, True, True),          # 0.90
    ({"english_quality": 3.5}, False, True),
    ({"content_error": .05}, True, True),
    ({"content_error": .2}, False, True),
    ({"content_error": .21}, False, False),
    ({"excluded_formalism": .05}, False, True),
    ({"instruction_fidelity": .79}, False, False),
])
def test_threshold_flags(over, strict, intermediate):
    assert select.flags(labelled(chat(1), **over)) == {"strict_pass": strict, "intermediate_pass": intermediate}


def test_overlap_removal():
    bench = labelled(chat(1, user="What is the capital of France? Answer in one word only please."))
    group_a = labelled(chat(2, user=LONG, reply="One."))
    group_b = labelled(chat(3, user=LONG, reply="Two."))  # same first prompt as group_a
    other = labelled(chat(4, user="Tell me a short story about a friendly dog in the park, please."))
    assert select.overlaps(bench, {text_key(bench.messages[0]["content"])})
    kept = select.isolate([group_a, group_b, other], set(), set())
    assert len(kept) == 2 and other in kept
    assert kept == select.isolate([other, group_b, group_a], set(), set())  # order-free, seeded


def pool(n, wave="original", start=0):
    subjects = ["arts", "history", "language"]
    return [labelled(chat(start + i, user=f"{LONG} number {i} {wave}", reply=f"Reply {i}."), wave=wave,
                     tokens=["short", "medium", "long"][i % 3], subject=subjects[i % 3],
                     reasoning_required=.9 if i % 4 == 0 else .1)
            for i in range(n)]


def test_cohorts_are_deterministic_and_proportional():
    rows = pool(90)
    first, report = select.cohort(rows, 30, "seed")
    again, _ = select.cohort(list(reversed(rows)), 30, "seed")
    assert [r.record_id for r in first] == [r.record_id for r in again]
    caps = Counter(select.features(r)["capability"] for r in first)
    assert caps["R-"] == round(30 * sum(i % 4 == 0 for i in range(90)) / 90)
    assert report["max_share_gap"] < .1


def test_membership_recipe():
    first = pool(40)
    second = pool(20, wave="extension", start=100)
    shared = labelled(chat(5, user="A different first prompt that is long enough to count.", reply="x"),
                      wave="extension")  # same source row as a first-wave chat: same family
    weak = labelled(chat(999), english_only=.5)
    chosen, counts = select.membership(first + second + [shared, weak], set(), set(), sizes=(10, 5, 5))
    splits = Counter(split for split, _ in chosen.values())
    assert splits == {"train": 55, "validation": 5} and weak.record_id not in chosen
    assert shared.record_id not in chosen and counts["strict"] == 61 and Counter(seg for _, seg in chosen.values())["long"] == 40


# Candidate sampling ----------------------------------------------------------------------

def test_allocation_minimum_and_remainders():
    assert sample.allocation({"a": 1, "b": 10, "c": 100}, 10) == {"a": 1, "b": 2, "c": 7}
    nested = sample.nested_quotas({"a": 50, "b": 150}, (20, 40))
    assert nested[20] == {"a": 6, "b": 14} and nested[40] == {"a": 12, "b": 28}


def test_draw_is_deterministic_and_records_inclusion():
    rows = [replace(chat(i, user=f"{LONG} {i}"), rendered_tokens=100 + 500 * (i % 3)) for i in range(60)]
    a = sample.draw(rows, (10, 20), seed="s", wave="w")
    b = sample.draw(list(reversed(rows)), (10, 20), seed="s", wave="w")
    assert [r.record_id for r in a] == [r.record_id for r in b] and len(a) == 20
    assert all(r.sample["wave"] == "w" and 0 < r.sample["inclusion_probability"] <= 1 for r in a)
    assert len({json.loads(r.sample["stratum"].split("|")[0])[2] for r in a[:10]}) == 3
