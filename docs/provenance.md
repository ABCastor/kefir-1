# Provenance

This page fixes the identities of the data, tokenizer, code and checkpoints behind Kefir 1. The code
architecture record ([code-record.md](code-record.md)) states which files in this tree are byte-identical to the code that ran.

## Data and tokenizer

| Item | Identity |
|---|---|
| Source data | `nvidia/Nemotron-ClimbMix`, revision `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`, licensed CC BY-NC 4.0 |
| Packed dataset | dataset run `moe-100b-superbpe-b64-dataset-003` in the public Hugging Face bucket `buckets/abcastor/kefir-1-pretraining-corpus`, prefix `run/moe-100b-superbpe-b64-dataset-003/` |
| Dataset manifest SHA-256 | `f9e7b77c347818a6379b1e450a604e739364fe99bd60ca7456dabbf3d534b97a` |
| Tokenizer | SuperBPE 8000 v2, `src/kefir/assets/superbpe_8000_v2.json` |
| `tokenizer.json` SHA-256 | `068e20ef8a16b1bff2a2a1d161e5a6f1cfd0935befa7c0ae734c9ec376d373bb` |
| Filter recipe | `english-prose-v4-scored-r2+adapter-2`; input hashes in [`data/pretraining/assets/FILTER-PROVENANCE.json`](../data/pretraining/assets/FILTER-PROVENANCE.json) |
| Cluster tree | [`data/pretraining/assets/cluster_tree.json`](../data/pretraining/assets/cluster_tree.json), SHA-256 `71734c7dafd16f414347ac498bc521ddfcc736487707d9bc1b703164ce333a41` |
| Corpus asset identities | [`data/pretraining/assets/manifest.json`](../data/pretraining/assets/manifest.json) (authority; also lists the original hashes of the two redacted assets) |

`FILTER-PROVENANCE.json` and `contract.json` omit environment-specific metadata; their historical SHA-256 values
stay in `data/pretraining/assets/manifest.json`. The tokenizer was trained earlier on
a separately filtered English seed.

## Pretraining

| Item | Identity |
|---|---|
| Training run | `moe-100b-superbpe-003` (steps 0–280,000) |
| Cooldown | C30, a fork of step 280,000 (steps 280,000–310,000) |
| Run 003 training-source tree SHA-256 (original, unpublished run tooling) | `8391db169f435fde92a98cf3a1dcec63bb34215bcc7adc28bd4926157776bc2c` (115 files) |
| Base weights at step 310,000, original inference export | SHA-256 `4dba9a25e925d55a23767ced2a67ab293225ce8bb5fe701645c910b27f88f71d` (as recorded in [`results/data/model.json`](../results/data/model.json)) |
| Base weights at step 310,000, released `.safetensors` file (576,329,232 bytes, exported from the step-310,000 trainer state) | SHA-256 `11ca59f45f8ebcfd019ed5a027198d6adb549502e3d118a031f4bec8931888ac` |

[`provenance/source-manifest.json`](provenance/source-manifest.json) lists the SHA-256 of every file of the original
run 003 training-source tree, under its original paths, with the original lock-file hashes. It is a historical
record: the code here was later reorganized under `src/kefir/`, so its paths and bytes differ from that tree, and
neither `pyproject.toml` nor `uv.lock` here describes the run 003 environment.

## Supervised fine-tuning

The SFT runs used the `src/kefir` runtime that is kept here byte for byte (90 files), plus the pairwise GDN-2
backend and the chat renderer (Renderer C, SHA-256 `d67c871f…`). Their hashes, and the seven entry and inference
files changed for this public tree, are listed in
[`tests/data/ran-runtime-manifest.json`](../tests/data/ran-runtime-manifest.json), which the test suite checks.
The 55,490 training and 512 validation conversations form the selected view under [abcastor/kefir-1-sft-conversations](https://huggingface.co/datasets/abcastor/kefir-1-sft-conversations). The [labelled SFT pool](https://huggingface.co/datasets/abcastor/kefir-1-sft-labelled-pool) records candidate scores and split membership.

| Released chat weights | Identity |
|---|---|
| Kefir 1 SFT-15k, flat batch 32 seed 17 | SHA-256 `106f44d44807f8fb12b8123137ae0de5421ec253935c9530c6c9d3bcc311cb96` |
| Kefir 1, 55,490 chats, added-System arm | SHA-256 `5d5bcb0d13b1df69c2625f336e724a4387126c2cef39e6d16a6aded2912758fe`: the trained file `753dacd688fbd836b1797cadcc4c4c6c7ad927c9a5ba76263528a6bd9aa57002` with `model_config` and tokenizer metadata added, tensors unchanged |

Rendering the selected conversations with `kefir sft render` gives 55,490 rows and 39,355,420 assistant targets,
the totals of the 55,490-chat run.


## Benchmarks

The frozen `kefir-bench/1` suite is identified by `src/kefir/bench/data/manifest.json` (split hashes) and its
upstream sources by `src/kefir/bench/data/source-manifest.json` (URLs, revisions, SHA-256, licences).
