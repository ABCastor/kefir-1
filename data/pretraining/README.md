# Kefir 1 pretraining corpus recipe

This directory contains the data preparation used for dataset run `moe-100b-superbpe-b64-dataset-003`. The source is NVIDIA Nemotron-ClimbMix at revision `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b` (source data: CC BY-NC 4.0). Preparing 100B tokens requires substantial storage and source downloads. Training can instead read the existing packed shards of that dataset run.

## What the builder does

1. `build_raw_parallel.py` reads the source's GPT-2 token IDs, removes a terminal GPT-2 EOD, decodes strict UTF-8, excludes programming cluster 11, applies the frozen v4 structural, fastText English, and student filters, checks exact/paragraph overlap with the frozen validation seed, and encodes accepted text with the **one** SuperBPE v2 artifact in `src/kefir/assets/`. It writes per-region `uint16` token streams and JSONL metadata, with EOD 7992 after each document. Exact deduplication is scoped to each region; run 003 did not replay cross-region deduplication.
2. `build_doc_index.py` makes an offset/cluster/score table for each raw region. `cluster_tree.py` derives the hierarchical cluster grouping; [the frozen tree](assets/cluster_tree.json) from all 2,100 original regions is bundled and is the default.
3. `strat_plan.py` and `pack_stratified.py` build 2,049-token records using same-cluster best-fit packing, assign 64 records per block by cluster and score/length strata, and globally shuffle blocks. The training reader uses each record's first 2,048 tokens as input and next 2,048 as labels. EOD padding remains a normal target, matching run 003. The packer writes schema-v2 shards, manifest, run contract, and optional Hugging Face bucket frontier using Kefir's `kefir.data` modules.

The historical raw-builder source hashes are recorded in [filter provenance](assets/FILTER-PROVENANCE.json) and the [asset manifest](assets/manifest.json). The scripts here have local paths and imports adapted to this repository; their own hashes are a **new** source identity. `filters_v4.py` is byte-identical to the historical source. The v1 normalization function was extracted from its old filter module; the old filter and the run 002 linear packer were not brought over. The original `seed-v2/validation-*.jsonl` is represented here by a deterministic 2.4 MB artifact containing exactly its 11,671 document IDs and 53,244 normalized paragraph hashes; source file hashes remain in the manifest. This preserves the overlap decision without copying 40 MB of seed text. The original builder's per-region transformations, v4 thresholds, record format, and run 003 packing/split algorithms remain the reference.

## Environment and assets

Use a separate Python 3.12 environment for corpus construction. The original filter stack pins NumPy 1.26.4, fastText and scikit-learn 1.5.2; it must not replace the trainer environment's NumPy version. Download the original `students.pkl` bundle (classifiers distilled from a decision model's labels) from [Kefir 1 Corpus Filters](https://huggingface.co/abcastor/kefir-1-corpus-filter), a Hugging Face model repo. From the repository root, download it, check its SHA-256 and verify the assets:

```bash
python3.12 -m venv data/pretraining/.venv
data/pretraining/.venv/bin/python -m pip install -r data/pretraining/requirements.txt
data/pretraining/.venv/bin/hf download abcastor/kefir-1-corpus-filter students.pkl \
  --revision main --local-dir data/pretraining/assets
printf '%s  %s\n' b8b8472278bc63a04f124b66fbefff9972a30d39ae98b13b541b772874135959 data/pretraining/assets/students.pkl | sha256sum -c -
python3 data/pretraining/fetch_fasttext.py
python3 data/pretraining/verify_assets.py
```

`fetch_fasttext.py` obtains the external `lid.176.bin` model from fastText's model host and verifies SHA-256 `7e69ec5451bc261cc7844e49e4792a85d7f09c06789ec800fc4a44aec362764e`. The 126 MB binary is not redistributed by this repository. The student classifier pickle `data/pretraining/assets/students.pkl` (SHA-256 `b8b8472278bc63a04f124b66fbefff9972a30d39ae98b13b541b772874135959`) is not bundled with this Git repository. The builder and `verify_assets.py` require the file and check its hash. Loading a pickle can execute code: use a trusted copy and verify the pinned hash before loading it directly with [`StudentGate`](student_gate.py). [`filters_v4_adapter.py`](filters_v4_adapter.py) calls `StudentGate` after the structural rules and fastText check. Source tokenizer IDs use GPT-2 solely at this decoding step. Model training, evaluation and generation use SuperBPE v2.

## Local recipe

Set `PYTHONPATH=src` so the separate corpus environment imports Kefir's shard and source reader modules. The commands below show a small local slice; remove the region limits and run partitioned workers for the full corpus. Raw workers read pinned NVIDIA source files over HTTP; no provider launcher is required.

```bash
PYTHONPATH=src data/pretraining/.venv/bin/python data/pretraining/build_raw_parallel.py \
  --worker 0 --workers 1 --regions 0-11 --max-regions 12 \
  --out data/pretraining/corpus-13b-v2/worker-0 --min-free-gib 1

PYTHONPATH=src data/pretraining/.venv/bin/python data/pretraining/build_doc_index.py \
  --root data/pretraining/corpus-13b-v2 --regions 12

PYTHONPATH=src data/pretraining/.venv/bin/python data/pretraining/pack_stratified.py \
  --root data/pretraining/corpus-13b-v2 --out data/pretraining/corpus-13b-v2/shards-strat \
  --regions 0-11 --max-blocks 200 --no-upload --run-id kefir-corpus-local-example
```

For the full recipe, build all 2,100 regions, index them with `--regions 2100`, then pack with `--regions 0-2099`, the frozen tree, and an explicit output directory. The historical worker contracts used a 24-worker partition and recorded `13,107,200,000` target tokens per worker; this was metadata, while each worker processed its assigned whole regions. To recreate the cluster grouping from the locally rebuilt raw corpus, run `cluster_tree.py --root data/pretraining/corpus-13b-v2 --docs-per-cluster 4000` and pass its output with `--cluster-tree`; a rebuilt tree may have a different hash if any input differs. Do not label a rebuilt dataset as the original run 003 unless its manifest and shard identities agree.

The historical worker bucket is not public. These commands use local raw worker files and do not require that bucket, private code, or provider accounts. `--no-upload` keeps shards local. Omit it only when intentionally publishing to a configured Hugging Face bucket with appropriate credentials; publication is a separate operational action.

## Verification boundary

`verify_assets.py` verifies both assets and the curated recipe sources, including the filter adapter,
student thresholds, and shared shard writer. `--sources-only` checks code without requiring the external
fastText binary. Raw building and packing perform the same source check before doing work. Intentional
recipe edits require updating the curated hashes and recording a new dataset identity; original source
hashes and historical corpus003 contracts remain unchanged.

New run-contract schema hashes identify the tokenizer by its SHA-256, without a local filesystem path.
Moving identical tokenizer bytes between machines therefore preserves the new contract identity.
Historical run003 metadata keeps its original schema hash; this change does not rewrite existing datasets.

The frozen builders have two resume limits. After a crash between the SQLite
commit and a region receipt write, the raw files can be complete while the
indexer cannot discover that region. Verify their hashes against the SQLite
receipt before recovering it. Existing index and packing-plan caches are not
fully bound to changed raw inputs or split settings; use a fresh output/cache
directory for a changed recipe. These historical scripts are SHA-pinned, so a
repair should be a separate tool or an explicitly versioned new recipe.

`PYTHONPATH=src data/pretraining/.venv/bin/python data/pretraining/test_recipe.py` checks fixed raw-region token/JSONL hashes and the frozen overlap contract. The original and curated indexers produced identical bytes for a 172-document local region. On 12 real source regions, the original and curated stratified packers each passed the schema-v2 verifier and produced byte-identical 20-block train and 4-block validation shards: SHA-256 `46e5660f64e02bfb1487c5016e25558ec318b2de9d727659136c10c2ccf3fa4c` and `da4edefcc70c7a9b7240baed1c8a0518630827d0f95bffd99baf3dd440b140b5`. This is a local equivalence check; no 100B rebuild or remote upload was performed during the distillation.

## Redacted provenance

`assets/contract.json` and `assets/FILTER-PROVENANCE.json` differ from the historical bytes only in names of a private qualification dataset, of the two build machines and, in `FILTER-PROVENANCE.json`, of the labelling teacher and its workspace paths; the original SHA-256 of each is kept under `redactions` in `assets/manifest.json`. The builder embeds `contract.json` in each raw-region contract, so a rebuilt raw contract hash differs from the historical one even when the token streams agree.
