# Resource acquisition

All commands run from the checkout root. [The machine-readable source manifest](../configs/resources.json) pins the exact Hugging Face dataset snapshots recorded by the experiments. The nine dataset revisions were checked against the Hub API on 2026-09-14. Qwen's four weight shards, model index, tokenizer and configuration were also checked against the pinned model revision using local SHA-256 hashes.

## Public downloads

```bash
fold2reason data download --resource model
fold2reason data download --resource general10-external
```

`general10-external` downloads nine datasets. FTB-Core is project-generated and comes from the archive below. To download one source, use a key such as `bbh`, `chembench`, `graphqa_easy`, `spatialviz_bench` or `vsi_bench`. `--list` prints the immutable revisions and paths without downloading. Network access and sufficient free disk space are required; the VSI video archives and their extracted copies are the largest resources. Authentication, if requested by a publisher, uses the local Hugging Face credential store. See the [official Hugging Face download CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).

Downloaded data are evaluation-only inputs for this workflow. Repository license terms do not replace the publishers' terms.

| Dataset key | Files consumed relative to its download directory | Selection |
|---|---|---|
| `bbh` | `*/test-*.parquet` | All supported task rows |
| `chembench` | `*/train-*.parquet` | All usable text benchmark rows; upstream names this split `train` |
| `chembench4k` | `test/*_benchmark.json` | All test rows |
| `graphqa_easy`, `graphqa_hard` | `data/test-00000-of-00001.parquet` | Full test split, canonical GraphQA scoring |
| `lab_bench` | `*/train-*.parquet` | Text-compatible questions; upstream split name is `train` |
| `scibench` | Top-level problem JSON files | All 580 problem rows; solution files are not extra examples |
| `spatialviz_bench` | `data/test-00000-of-00001.parquet`, `SpatialViz_Bench_images/` | 1,180 real-image questions |
| `vsi_bench` | `test.jsonl`, `test_debiased.parquet`, `arkitscenes/`, `scannet/`, `scannetpp/` | 5,130 real-video questions; debiased IDs retain the scorer's existing role |

The files already match the evaluator schemas; do not re-export, concatenate other splits, or renumber rows. The downloader extracts the contents of `SpatialViz_Bench_images.zip` into `SpatialViz_Bench_images/` and the three VSI zip files into their dataset root. It validates existing extracted files and refuses to overwrite different files. `--skip-extract` downloads the archives only.

The official sources are linked individually in the [README](../README.md#models-and-evaluation-data). The original BBH task collection is also available at [BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard); the recorded Parquet snapshot is used here to keep the loader format and row order fixed.

## Project resource archives

This release provides **six data archives, the original frozen geometry decoder, and the Qwen3.5-9B main experiment's three-seed LoRA/workspace checkpoints**. Base-model weights are excluded. Archive files and checksums are recorded in [release_assets.json](release_assets.json). Google Drive URLs are `null` until the owner supplies the upload links.

| Archive | Use | Required? |
|---|---|---|
| `fold2reason-training-core-v1.tar.gz` | Main 1,000 train / 100 dev / 100 held-out protein records, geometry cache, FoldingCorpus cache and human-readable supervision | Training from scratch |
| `fold2reason-frozen-decoder-heads-v1.tar.gz` | Original coordinate and distogram heads plus configuration; no adapter or tokenizer | Full-method training and original structural readout |
| `fold2reason-qwen35-9b-main-lora-workspace-3seeds-v1.tar.gz` | Main experiment only: seeds 20260729, 20260803, 20260804; fixed epoch 3 / step 375 | Evaluate the main result without retraining |
| `fold2reason-foldbench334-v1.tar.gz` | The exact 334-target cache, target IDs and BB4Q10 input/target records | Structural evaluation |
| `fold2reason-ftb-core-v1.tar.gz` | All four FTB-Core splits and original split hashes; evaluation uses test + test_ood only | Full General-10 |
| `fold2reason-scaling-seed-20260729-v1.tar.gz` | Frozen caches at 50 / 100 / 250 / 500 / 1,000 / 2,000 / 4,000 proteins | Data-scaling experiments |
| `fold2reason-scaling-seed-20260803-v1.tar.gz` | Same sizes, second seed's selected sets and order | Data-scaling experiments |
| `fold2reason-scaling-seed-20260804-v1.tar.gz` | Same sizes, third seed's selected sets and order | Data-scaling experiments |

Verify each downloaded archive against the manifest, then use the installer:

```bash
fold2reason data install-archive \
  --archive downloads/fold2reason-training-core-v1.tar.gz
```

Repeat for the archives needed by your experiment. The installer checks the archive SHA-256, rejects unsafe entries and refuses to overwrite existing files. It places files under `data/` and `artifacts/` in the current checkout. Internal manifests under `artifacts/resource_manifests/` record every payload file's source/release hash. Private provenance paths were replaced by descriptive `resource://source/...` references; tensor payloads and sample order are unchanged. These references document origin, not remote downloads.

For the main experiment, the installed paths match [reproduction](reproduction.md): `artifacts/cache/foldingcorpus.pt`, `artifacts/cache/geometry.pt`, `artifacts/cache/foldbench334.pt`, and `artifacts/decoder/final`. Use the same main cache for all three training seeds. Install the decoder archive and skip decoder retraining, then train the full model or Pure-LoRA baseline in section 4. The decoder archive preserves the original coordinate/distogram heads byte-for-byte; it is not a standalone language-model/adapter checkpoint. It supports the full-method trainer and workspace FoldBench evaluator through `--frozen-decoder-checkpoint`. General-10 inference itself uses only the language model and trained adapter.

For fixed-epoch scaling, use `configs/train/scaling.toml`, the corresponding `--seed`, and `--cache artifacts/cache/scaling/seed-SEED/dNNNN_q12.pt`. Each seed has its own frozen nested selections; do not substitute a newly sampled subset. No benchmark is used to choose checkpoints.

The main checkpoint archive installs to `artifacts/pretrained/qwen35-9b-main/seed-SEED/`. Each seed has `adapter/adapter_model.safetensors`, `adapter/adapter_config.json`, `workspace.pt` and `workspace_config.json`. These are the paper's full FoldingCorpus + geometry result, with 1,000 training proteins; the legacy code calls the arm `m3_no_retrieval` because the separate retrieval objective is disabled. The archive excludes scaling/control checkpoints, other models, intermediate steps and optimizer state. Keep each adapter paired with its same-seed workspace. See [pretrained evaluation](evaluation.md#evaluate-the-released-main-checkpoints).

## FoldBench and training provenance

[Official FoldBench](https://github.com/BEAM-Labs/FoldBench) provides the upstream target definitions and MIT-licensed code. The paper's structural evaluation uses the local **processed 334-protein snapshot** in the project archive, including its specific sequence/MSA/template rendering and tokenizer cache. An arbitrary current upstream checkout is not a substitute for that snapshot. Use the target manifest to identify the evaluated proteins and the cache to reproduce their inputs.

The training archive is a selected **OpenFold high-confidence distillation snapshot**, with MGYP identifiers and predicted coordinate targets; it is not described as a collection of experimentally determined PDB structures. The owner's original acquisition record and distillation-corpus redistribution terms still need to be attached before public release. [OpenProteinSet](https://registry.opendata.aws/openfold/) is a related CC BY 4.0 resource, but has not been established as the source of every predicted structure in this snapshot. [PDB archive data](https://www.wwpdb.org/about/usage-policies) have their separate CC0 terms.

Archive notices retain Qwen and FoldBench licenses. The project's own data/decoder/adapter release license remains an owner decision. Nine third-party General-10 datasets are downloaded from their publishers rather than re-hosted in the project bundles.

## Verify and build visual features

```bash
fold2reason data verify --benchmark-root data/benchmarks
fold2reason data visual \
  --model models/Qwen3.5-9B \
  --spatialviz-root data/benchmarks/data/external/spatialviz_bench \
  --vsi-root data/benchmarks/data/external/vsi_bench \
  --num-video-frames 32 \
  --output artifacts/cache/visual
```

The first command checks the full suite's 76,725 example IDs/counts and media paths, without model inference. The second command runs GPU preprocessing to build the model-specific visual features. These generated caches are not included in the upload bundles. Continue with [evaluation](evaluation.md) after preprocessing.
