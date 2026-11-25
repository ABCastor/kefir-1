# Third-party material

The code in this repository is licensed under the MIT licence ([LICENSE](LICENSE)). The Kefir 1 weights are
licensed CC BY-NC 4.0 (non-commercial). The items below are not covered by the MIT licence, or are not shipped at all.

## Shipped here under another licence

| Path | What it is | Licence |
|---|---|---|
| `src/kefir/assets/thematic_panels_v1/*.npz` | 12 validation panels: at most 32 windows of 2,048 token IDs each, tokenized excerpts of held-out text from [NVIDIA Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) (revision `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`) | CC BY-NC 4.0, as the source dataset |
| `data/pretraining/assets/validation-overlap.json.gz`, `work-plan.json`, `cluster_tree.json` | document hashes, region plan and cluster statistics computed from the same source; no source text | MIT (facts about the source, not its text) |
| `src/kefir/assets/superbpe_8000_v2.json` | the 8,000-token SuperBPE vocabulary, trained by the authors on about 6 GB of filtered English prose from NVIDIA Nemotron-ClimbMix (revision `5eaa64b9`, CC BY-NC 4.0; the vocabulary holds no source text); code in `data/tokenizer/` | MIT |

## Not shipped; obtain from upstream

| Item | Where | Licence |
|---|---|---|
| Pretraining source text | `nvidia/Nemotron-ClimbMix` on Hugging Face | CC BY-NC 4.0 (NVIDIA's dataset licence; rights in the underlying web documents were not assessed, see the upstream card) |
| fastText language identifier `lid.176.bin` | fetched by `data/pretraining/fetch_fasttext.py`, SHA-256 checked; used, not redistributed, by the corpus filter (library `fasttext-wheel==0.9.2`, `data/pretraining/requirements.txt`) and by the fine-tuning language pre-filter (library version not recorded) | model CC BY-SA 3.0; library MIT |
| Lingua language detector | used, not redistributed, by the fine-tuning language pre-filter; version not recorded | Apache 2.0 |
| Corpus student classifiers `data/pretraining/assets/students.pkl` | [`students.pkl`](https://huggingface.co/abcastor/kefir-1-corpus-filter/blob/main/students.pkl) in the Hugging Face model repo `abcastor/kefir-1-corpus-filter`. Use a trusted copy and verify its pinned SHA-256 (`b8b8472278bc63a04f124b66fbefff9972a30d39ae98b13b541b772874135959`) before loading (trained on CC BY-NC 4.0 text). | CC BY-NC 4.0 |
| Benchmark items | rebuilt by `kefir bench build` from BLiMP, ARC-Easy and PopQA at pinned revisions (`src/kefir/bench/data/source-manifest.json`) | CC BY 4.0, CC BY-SA 4.0, MIT |

## Fine-tuning conversations

The supervised fine-tuning data is a filtered selection of these public datasets; each conversation keeps the
licence of its source. An open-weights decision model labelled the candidates; its labels are released in the
[labelled pool](https://huggingface.co/datasets/abcastor/kefir-1-sft-labelled-pool).

| Source | Licence declared on its dataset card |
|---|---|
| `HuggingFaceTB/smol-smoltalk` | Apache 2.0 |
| `HuggingFaceTB/smoltalk2` | Apache 2.0 for its new parts; included datasets keep their own terms |
| `HuggingFaceH4/ultrachat_200k` | MIT |
| `OpenAssistant/oasst1`, `OpenAssistant/oasst2` | Apache 2.0 |
| `allenai/tulu-3-sft-personas-instruction-following` | ODC-BY |
| `databricks/databricks-dolly-15k` | CC BY-SA 3.0 |
| `HuggingFaceH4/no_robots` | CC BY-NC 4.0 |

## Weights

The released weights are licensed **CC BY-NC 4.0** (non-commercial), as their pretraining source text; each model
card states the same. They were also fine-tuned on conversations from databricks-dolly-15k (CC BY-SA 3.0): 569 Dolly rows are in the
training data of Kefir 1; the Dolly rows in the training data of Kefir 1 SFT-15k were not counted.
The conversations retain their source licences when redistributed.
