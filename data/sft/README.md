# Kefir 1 fine-tuning data recipe

This directory shows how the supervised fine-tuning (SFT) conversations of Kefir 1 were collected, labelled by a decision model, filtered and selected. It is the post-training counterpart of the [pretraining corpus recipe](../pretraining/README.md). Its output is the published [labelled SFT pool](https://huggingface.co/datasets/abcastor/kefir-1-sft-labelled-pool) (200,000 labelled candidates) and its membership: 55,490 training and 512 validation conversations ([selected view](https://huggingface.co/datasets/abcastor/kefir-1-sft-conversations)). The labels came from one open-weights decision model; any judge that answers the same ten questions can take its place (see [Plugging in a judge](#plugging-in-a-judge)).

## What the recipe does

Every step reads and writes JSON Lines of one record type, `Record` in [`records.py`](sft_data/records.py): a whole conversation (role/content messages), its source pointer, its isolation groups and, as steps add them, its rendered length, sampling stratum and judge answers. Text is normalised once, on entry (Unicode NFC, `\n` line endings, an optional first System message, then User/Assistant alternation ending with an Assistant reply). Nothing later edits text; a conversation that does not fit a rule is dropped whole.

1. **Sources** ([`sources.py`](sft_data/sources.py)). Eight public datasets, each read from files of one pinned revision:

   | Source | Revision | Training files | Held-out files (isolation only) |
   |---|---|---|---|
   | `HuggingFaceTB/smol-smoltalk` | `f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc` | 4 `train` parquet | `test` |
   | `HuggingFaceH4/ultrachat_200k` | `8049631c405ae6576f93f445c6b8166f76f5505a` | 3 `train_sft` parquet | `test_gen`, `test_sft` |
   | `allenai/tulu-3-sft-personas-instruction-following` | `fe0c7d350c9b4542b8d829a6f1daa1c259f0ba0e` | `train` | none |
   | `HuggingFaceTB/smoltalk2` (`SFT`, SystemChats, no thinking) | `fc6cc2103c066455aade5d7fbb346039ae36ca5e` | one parquet | none |
   | `OpenAssistant/oasst2` | `179dd21fc55192153d94adb0e0ce8f69e222bf75` | `train` | `validation` |
   | `databricks/databricks-dolly-15k` | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | the JSONL file | none |
   | `OpenAssistant/oasst1` | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | `train` | `validation` |
   | `HuggingFaceH4/no_robots` | `e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b` | `train` | `test` |

   Flat chat rows become one conversation each. Dolly's instruction and context become one User message (`instruction`, a blank line, `Context:` and the context). SmolTalk2's custom instructions become the System message. OpenAssistant releases are message trees and are **flattened**: each maximal valid path from a root prompt to an Assistant reply is one conversation, and shorter paths that are prefixes of a kept path are not separate rows. A path stops at the first message that is not English, is deleted, failed review or breaks the role order. Each conversation gets isolation groups: its source row, native thread, message, prompt and conversation IDs where the source has them, and its first User prompt when it has at least six words and 40 characters.

2. **Length cap and source domain** (`prepare`). Each conversation is rendered with Renderer C (`src/kefir/sft/renderer.py`, SHA-256 `d67c871f…`). Conversations longer than **2,048 rendered tokens** are excluded, never truncated. For a threaded conversation that is too long, the longest prefix ending in an Assistant reply that fits is kept instead, with the groups of the full path. No Robots rows of the native `Coding` category are excluded.

3. **Deduplication and held-out isolation** ([`dedup.py`](sft_data/dedup.py)). Conversations sharing an isolation group, with identical content or near-duplicate text (Jaccard ≥ 0.9 over distinct case-folded 5-word shingles, exactly verified) form one family. A family containing any held-out conversation is removed. Of identical conversations one is kept, and it inherits the others' groups. Two different near-duplicate survivors are both rejected unless they are branches of one OpenAssistant tree or one is a whole-message prefix of the other. Historically this left **724,938** candidates.

4. **Scope and language pre-filters** ([`prefilter.py`](sft_data/prefilter.py), versions `scope-prefilter-v2.1` and `language-scope-v1`). Scope rules are presence tests over every message of every role: non-Latin letters, formal logic and set symbols, LaTeX, markup, SQL, JSON objects, program code, shell commands and symbolic variable relations. Ordinary Markdown and numeric arithmetic pass. On the survivors, every message and paragraph of at least 12 words is checked with fastText `lid.176.bin` (SHA-256 `7e69ec54…`); a non-English label with score ≥ 0.8 is rechecked with Lingua over all its languages. A conversation is excluded only when both name the same non-English language with fastText ≥ 0.95, Lingua ≥ 0.98 and a Lingua margin ≥ 0.8. Historically 115,171 conversations were excluded by scope and 2,631 by language, leaving **607,136**. Passing these filters does not certify English or the absence of code; the judge questions measure that.

5. **Stratified candidate sampling** ([`sample.py`](sft_data/sample.py)). Base cells are source × Assistant-turn bin (1, 2–3, 4–6, 7+) × rendered-length bin (≤ 512, ≤ 1,024, > 1,024) × System presence. Cells with at least 40 members are split by available source metadata (native category and source, instruction constraints, OpenAssistant ratings, origins of merged duplicates, longest message length), keeping children of at least 20 and a residual, at most 6 splits per dimension and 600 cells. Each cell receives min(2, size) conversations, the rest proportionally by largest remainders, and within a cell the lowest seeded hashes are drawn. Each sampled record keeps its stratum and inclusion probability. With nested milestones each increment is allocated from what remains, so every milestone is itself a stratified sample. Historically two waves of 100,000 were labelled: `original` (a 5,000 pilot, a 5,000 throughput sample and 90,000 at milestones 15k/40k/65k/90k) and `extension` (100,000 at 25k/50k/75k/100k from the 506,312 not yet sampled).

6. **The ten judge questions** ([`questions.py`](sft_data/questions.py), [`questions.json`](sft_data/questions.json)). The judge reads each conversation as plain text with a fixed preamble and answers ten questions separately: `english_only`, `excluded_formalism`, `english_quality`, `instruction_fidelity`, `content_error`, `task`, `subject`, `reasoning_required`, `context_required` and `factual_uncertainty`. The question text, criteria and preamble are the exact frozen set (SHA-256 `3a0c8f02…`, the pool's `schema_hash`). Yes/no questions return the probability of yes. `english_quality` returns an expected level on a 0–4 rubric. `task` and `subject` return one option.

7. **Score normalisation.** The eight numeric scores are in 0–1: the probability of yes, or the English-quality level divided by 4. Task and subject stay categorical.

8. **Filter thresholds** ([`selection.py`](sft_data/selection.py)). All five must hold:

   | Score | Strict | Intermediate |
   |---|---:|---:|
   | `english_only` | ≥ 0.99 | ≥ 0.95 |
   | `english_quality` | ≥ 0.90 | ≥ 0.80 |
   | `instruction_fidelity` | ≥ 0.95 | ≥ 0.80 |
   | `content_error` | ≤ 0.05 | ≤ 0.20 |
   | `excluded_formalism` | ≤ 0.01 | ≤ 0.05 |

   Of the 200,000, 56,661 pass the strict filter (28,058 in the first wave, 28,603 in the second) and 116,172 the intermediate one. The training data uses the strict filter.

9. **Overlap removal.** A strict conversation is dropped if any of its User messages equals (case- and whitespace-insensitively) a message of the 1,022 frozen benchmark cases; for the second wave, also a prompt of the 80-prompt open-generation panel. [`assets/overlap-keys.json`](sft_data/assets/overlap-keys.json) holds only SHA-256 hashes of those texts. Then families are thinned: conversations are visited in a seeded hash order and one is kept only if it shares no isolation group and no first User message with a conversation already kept. The second wave is thinned against everything kept from the first. This left 27,854 independent conversations in the first wave and 28,148 in the second (3 benchmark matches and 452 shared families or first prompts removed).

10. **Stratified training cohorts and the 512 validation split.** From the first-wave independent pool, cohorts are drawn with Hamilton allocation by source × capability (reasoning and context scores ≥ 0.8), and the best of 64 seeded draws for the margins of source, task, subject, length bin, turn bin, System presence and capability. In order: **10,000** training conversations (seed `kefir-sft-10k-strict-v1:train`), the **512** validation conversations from the rest (`…:validation`) and **5,000** continuation conversations (`kefir-sft-continuation-5k-v1`). The long continuation then takes every remaining independent conversation: 12,342 from the first wave and 28,148 from the second, **40,490** in all. Training membership is 10,000 + 5,000 + 40,490 = **55,490**; the other **143,998** labelled candidates are unused.

The training order of each segment and the System text added to part of the long continuation are training-time choices and are not part of this directory ([docs/method.md](../../docs/method.md)).

## Environment

Use the pretraining corpus environment or any Python 3.12 environment. The package is `sft_data/` in this directory: run it from the repository root with `PYTHONPATH=data/sft python -m sft_data <command>`. `records`, `questions`, `selection`, `dedup`, `sample` and `judge` need only the standard library. Reading Parquet needs `pyarrow`; `prepare` also needs the repository's tokenizer (`PYTHONPATH=src:data/sft` and `tokenizers`); `prefilter` needs `fasttext-wheel` and `lingua-language-detector` (the historical language audit records Lingua 2.1.1; the fastText package version of the historical run was not recorded in the sources read for this port, and [`data/pretraining/requirements.txt`](../pretraining/requirements.txt) pins `fasttext-wheel==0.9.2`). Get `lid.176.bin` with `python3 data/pretraining/fetch_fasttext.py`; it is not redistributed.

## Commands

From the repository root. Source files are downloaded from Hugging Face at the pinned revisions; `sources` prints the exact `hf download` commands.

```bash
export PYTHONPATH=data/sft
python -m sft_data sources                                   # download commands into data/sft/sources
PYTHONPATH=src:data/sft python -m sft_data prepare --root data/sft/sources --out data/sft/work/candidates.jsonl
python -m sft_data prefilter data/sft/work/candidates.jsonl \
  --fasttext data/pretraining/assets/lid.176.bin --out data/sft/work/retained.jsonl
python -m sft_data sample data/sft/work/retained.jsonl --milestones 25000,50000,75000,100000 \
  --seed my-seed --wave original --out data/sft/work/wave-1.jsonl
python -m sft_data label data/sft/work/wave-1.jsonl --base-url http://localhost:8000/v1 --model my-judge \
  --out data/sft/work/wave-1-labelled.jsonl
python -m sft_data select data/sft/work/wave-1-labelled.jsonl data/sft/work/wave-2-labelled.jsonl \
  --first-wave original --out data/sft/work/membership.json
```

A second wave is sampled with `--exclude data/sft/work/wave-1.jsonl`. Both `data/sft/sources/` and `data/sft/work/` are ignored by Git. Every command refuses an existing output file and prints a JSON report of its counts (`--report` also saves it).

The historical seeds were `kefir-rich-ten-v2` (pilot), `kefir-overnight-ten-v1` and `kefir-extension-ten-v1`. The historical candidate stages ran as remote jobs over sharded files with a disk-backed deduplication index; the in-memory `prepare` above follows the same rules but has only been run on small fixtures, not on the full sources. A rebuild from today's files is a new dataset unless its record IDs match the published pool.

## Plugging in a judge

A judge is any object with one method:

```python
class Judge(Protocol):
    def answer(self, state: str, questions: dict) -> dict: ...
```

`state` is `questions.state(messages)`; the result maps each question name to an answer object (`{"type": "noul", "noul": p}`, `{"type": "score", "score": s}`, `{"type": "choice", "choice": k}`), validated by `questions.parse_answers`. [`judge.py`](sft_data/judge.py) has one reference adapter, `ChatJudge`, for any OpenAI-compatible `/chat/completions` endpoint, at the base URL and model name you pass (`--base-url`, `--model`, both required, no default): a local server (vLLM, llama.cpp, Ollama) or a remote service of your choice. It sends one request per question at temperature 0, asks for probabilities over the question's options and converts them to answer objects. A key, if needed, is read from the environment variable named by `--api-key-env`; there are no retries.

Probabilities stated by a chat model are not those of the decision model that labelled Kefir 1, so the thresholds above may select a different share with another judge. Check the selected share and a sample of decisions before reusing them.

## Verification

`python -m pytest data/sft -q`, from the repository root, runs CPU tests on tiny synthetic conversations: answer parsing, score mapping, threshold flags, overlap removal, deterministic cohorts and sampling, threaded flattening, pre-filters and deduplication.

`PYTHONPATH=data/sft python -m sft_data check <parquet parts>` recomputes, from each published pool row, its `record_id` and `content_hash`, its eight scores and both filter flags, and reports any disagreement. On part `part-00000.parquet` (10,000 rows) every field agreed. In a separate check made while porting, the judge input built by `questions.state` with the frozen questions reproduced all 10,000 published `request_hash` values of that part. With all 20 parts and `--membership`, the check also reruns overlap removal and cohort selection and compares membership with the published one. The pool does not store groups inherited from merged duplicates (a few hundred rows), so the rerun can differ slightly from the original; this full rerun has not been performed yet.
