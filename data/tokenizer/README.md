# Kefir 1 tokenizer recipe

This directory contains the code that trained the Kefir 1 tokenizer, the 8,000-ID SuperBPE vocabulary bundled as
`src/kefir/assets/superbpe_8000_v2.json` (SHA-256 `068e20ef8a16b1bff2a2a1d161e5a6f1cfd0935befa7c0ae734c9ec376d373bb`)
and published as [`abcastor/superbpe-8k-english`](https://huggingface.co/abcastor/superbpe-8k-english).

## What SuperBPE is

SuperBPE ([Liu et al., 2025](https://arxiv.org/abs/2503.13423)) is byte-level BPE trained in two stages. The first
stage learns ordinary subword tokens, which never cross a space. The second stage lifts that boundary, so frequent
multi-word strings such as ` of the` or ` by the way` become single tokens ("superwords").

## The two stages

1. **Subwords.** Byte-level BPE with the GPT-4o pre-tokenization regex below, up to 7,192 entries: the 243 bytes
   that can occur in valid UTF-8 plus 6,949 merges. The 13 bytes that cannot (`0xC0`, `0xC1`, `0xF5`–`0xFF`) are
   not in the vocabulary.
2. **Superwords.** Starting from the stage-1 model, 800 supermerges are learned with the whitespace boundary lifted
   (IDs 7,192–7,991). The switch point is 7,192 of 7,992 learned entries, 90%, the setting the SuperBPE paper found
   best downstream rather than the one that compresses most.

Eight special tokens follow. The embedding table of Kefir 1 is padded to 8,192 rows.

| Parameter | Value |
|---|---|
| Library | `boundlessbpe` 0.1.0, built from [fastboundlessbpe](https://github.com/kensho-technologies/fastboundlessbpe) commit `c2dce3131093aa4c7fc595c91b15a861dea3094c` (Apache-2.0), Python 3.12 |
| Stage 1 | `FastBpeTrainer(main_regex=GPT4O_REGEX).train(tau=1.1, vocab_size=7192, recalc=0, blowup=False, max_bytes=11_000_000_000)` |
| Stage 2 | `FastSuperBpeTrainer(main_regex=GPT4O_REGEX).train(vocab_size=800, recalc=0, word_model_file=word_final.model, max_bytes=1_100_000_000)` |
| `tau=1.1` | above 1, so PickyBPE token deletion is off: plain BPE |
| `recalc=0` | the library's periodic recount check is off |
| Special tokens | 7992 `<\|endoftext\|>` (end of document), 7993 `<think>`, 7994 `</think>`, 7995 `<answer>`, 7996–7999 `<\|reserved_0\|>`…`<\|reserved_3\|>` |
| Vocabulary | 8,000 IDs: 7,192 subword + 800 superword + 8 special; 7,749 merges |

Stage-1 pre-tokenization (`boundlessbpe.GPT4O_REGEX`, the parts joined with `|`):

```text
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
\p{N}{1,3}
 ?[^\s\p{L}\p{N}]+[\r\n/]*
\s*[\r\n]+
\s+(?!\S)
\s+
```

## Export to `tokenizer.json`

[`export_hf.py`](export_hf.py) writes the trained `.model` as a Hugging Face `tokenizers` file in the layout of the
public SuperBPE release: a BPE model whose merges are the stage-1 merges followed by the supermerges, in training
order; pre-tokenizer `Digits(individual_digits=False)`, then a split of digit runs into groups of three from the
right, then `ByteLevel(add_prefix_space=False, use_regex=False)`; `ByteLevel` decoder; no normalizer and no unknown
token. Without the byte-level regex, supermerges are reachable at inference.

The exported file and the training library segment the same text identically on 52.2% of 5,000 held-out documents,
with zero round-trip failures and 4.351 against 4.356 bytes per token. The reference applies supermerges only between
whole pre-tokens; Hugging Face BPE can also apply them across word fragments. `tokenizer.json` is the authority: the
corpus was encoded with it. Kefir encodes text with the IDs below 7992 only and appends 7992 after each document, so
ordinary text can never produce a control ID.

## How the seed was built

The training text is a separately filtered English seed, not the pretraining corpus, which later used a stricter
filter. Its source is [NVIDIA Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) at
revision `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b` (CC BY-NC 4.0): about 10 GB of documents decoded from the
GPT-2 token IDs of an earlier 10B-token training mixture drawn from that source (programming cluster 11 excluded),
documents split on GPT-2 end-of-text. A separate held-out portion was used only for validation.

[`prepare_seed.py`](prepare_seed.py) then applies, per document:

- the `english-prose-v1` rules in [`filters.py`](filters.py): whole documents with code, markup, LaTeX, symbolic
  maths, equations, programming topics, PII patterns, special-token literals or control characters are rejected,
  as are short (< 40 words), letter-poor, non-Latin, link-heavy or repetitive documents, and any document with a
  2,000-character chunk that fastText `lid.176.bin` does not label English with probability ≥ 0.90;
- exact deduplication on the SHA-256 of the NFKC-casefolded, whitespace-collapsed text;
- removal of training documents that share a paragraph of 40+ words with a validation document, and at most 1,000
  validation documents per keyword topic;
- the `english-prose-v2` refinement: documents with function notation, inline assignments, numeric formulas or
  ASCII logic operators are dropped.

Stage 1 trained on every surviving training document: 2,219,913 documents, 5,848,388,156 UTF-8 bytes. Stage 2
trained on the documents whose ID satisfies `int(id[:8], 16) % 5 == 0`, in the same order, up to 10⁹ bytes: 377,370
documents, 999,999,814 bytes. The 11,671 validation documents were in neither stage. The two files are lines of
`{"text": ...}`; their SHA-256 values are in [`manifest.json`](manifest.json). The seed text is not redistributed.

## Environment

Use Python 3.12 and a separate environment. `boundlessbpe` is built from source, so a Rust toolchain is needed.

```bash
python3.12 -m venv data/tokenizer/.venv
data/tokenizer/.venv/bin/python -m pip install -r data/tokenizer/requirements.txt
```

`prepare_seed.py` also needs `lid.176.bin`; [`fetch_fasttext.py`](../pretraining/fetch_fasttext.py) downloads and
checks it.

## Verify the bundled tokenizer

```bash
python data/tokenizer/verify_tokenizer.py
cd data/tokenizer && python -m unittest test_tokenizer
```

`verify_tokenizer.py` needs only `tokenizers`. It checks the SHA-256, the layout (8,000 IDs, 7,749 merges, special
tokens at 7992–7999, 800 superword entries) and encode/decode round trips on ASCII, control characters, digits,
whitespace, accented letters, CJK, symbols and emoji. With `--model superbpe_8000_final.model`, downloaded from the
tokenizer repository on Hugging Face, it also re-exports that producer model and checks that the result is
byte-identical to the bundled file.

The test checks the bundled file, the seed selection rules, and, when `boundlessbpe` is installed, a two-stage fit
on generated text whose export must have the published format: same pre-tokenizer, decoder and model settings,
same byte alphabet in the same ID order, stage-1 merges followed by the supermerges, special tokens after them.

## Retrain

On a small sample, in seconds, with a smaller vocabulary (bring any JSONL files with a `text` field):

```bash
cd data/tokenizer
.venv/bin/python train_tokenizer.py --stage1 sample1.jsonl --stage2 sample2.jsonl --out work/small \
  --stage1-vocab 400 --supermerges 40 --parity heldout.jsonl --parity-docs 200
```

The full recipe, with the published parameters as defaults:

```bash
.venv/bin/python prepare_seed.py --lid ../pretraining/assets/lid.176.bin \
  --validation heldout/*.jsonl --train train/*.jsonl --out work/seed
.venv/bin/python train_tokenizer.py --stage1 work/seed/stage1.jsonl --stage2 work/seed/stage2.jsonl \
  --out work/tok --parity work/seed/validation.jsonl
```

Each run writes `word_final.model`, `superbpe_8000_final.model`, `tokenizer.json` and `train-receipt.json`. The
training fails if a stage stops before its target size or a special token is not where expected; `--parity` adds
the original gate (no round-trip failure, token counts within 0.5% of the reference). A later refit of the same
recipe on a lowercased copy of the seed took 19 minutes on one CPU core.

## Verification boundary

Verified: the bundled file's hash and round trips; that `export_hf.py` turns the published
`superbpe_8000_final.model` into a byte-identical `tokenizer.json`; that the training calls produce that format on
a small sample, deterministically. Not verified here: a full retraining. The seed cannot be rebuilt from this
directory alone: the 10B-token mixture it was decoded from is not published, and a rebuild from ClimbMix with
another selection would give a different, if similar, vocabulary. Do not call a retrained file the Kefir 1
tokenizer unless its SHA-256 is `068e20ef…d373bb`.
