# Evaluation

## Evaluate the released main checkpoints

Install the Qwen3.5-9B main experiment archive to evaluate the existing fixed epoch-3 / step-375 models:

```bash
fold2reason data install-archive \
  --archive downloads/fold2reason-qwen35-9b-main-lora-workspace-3seeds-v1.tar.gz
```

The three seeds are `20260729`, `20260803`, and `20260804`. In the commands below, replace `artifacts/runs/full/seed-20260729/final` with `artifacts/pretrained/qwen35-9b-main/seed-20260729`; the `adapter/` subdirectory stays the same. Repeat with the other seeds and separate output directories.

General-10 uses the base model plus the selected seed's LoRA. FoldBench additionally uses the same seed's workspace and the separately installed original frozen decoder at `artifacts/decoder/final`. Tokenizer files come from the pinned official base model. The archive includes no base weights or optimizer state. Its internal `checkpoint_manifest.json` records source checkpoint paths, per-file hashes, workspace tensor hashes and fixed endpoint metadata.

## Structural readout

```bash
torchrun --standalone --nproc_per_node=4 -m fold2reason evaluate foldbench \
  --model models/Qwen3.5-9B \
  --checkpoint artifacts/runs/full/seed-20260729/final \
  --frozen-decoder-checkpoint artifacts/decoder/final \
  --cache artifacts/cache/foldbench334.pt \
  --label full-seed-20260729 \
  --output artifacts/eval/foldbench/full-seed-20260729
```

This computes the repository's coordinate-based structural readouts, including lDDT-Cα, contact F1, and its alignment-based TM-score implementation. It is not a wrapper around an external TM-align executable. The reference cache contains 334 monomer targets; check the completed evaluator output against the cache IDs. `--max-examples 0` is the default full-cache setting.

Held-out FoldingCorpus label accuracy for a Pure-LoRA checkpoint is available through `fold2reason evaluate corpus --help`. Use the same tokenizer/cache and the corresponding JSONL records.

## Full General-10

| Group | Benchmarks | Command |
|---|---|---|
| Spatial | FTB-Core | `evaluate ftb` |
| Spatial | SpatialViz, VSI | `evaluate spatial` |
| Graph | GraphQA Easy, GraphQA Hard | `evaluate general10-text` |
| Scientific | ChemBench, ChemBench4K, Lab-Bench, SciBench | `evaluate general10-text` |
| Mixed | BBH | `evaluate general10-text` |

The suite has ten benchmarks. The text command covers **seven**, not all ten. The public text recipe explicitly disables per-dataset truncation; pass no mini selection manifest for full evaluation.

```bash
torchrun --standalone --nproc_per_node=4 -m fold2reason evaluate general10-text \
  --config configs/eval/general10_text.toml \
  --model models/Qwen3.5-9B \
  --adapter artifacts/runs/full/seed-20260729/final/adapter \
  --general-benchs-root data/benchmarks \
  --output artifacts/eval/text/seed-20260729

torchrun --standalone --nproc_per_node=4 -m fold2reason evaluate ftb \
  --config configs/eval/ftb.toml \
  --model models/Qwen3.5-9B \
  --adapter artifacts/runs/full/seed-20260729/final/adapter \
  --benchmark-root data/benchmarks/data/generated/ftb_core/v1 \
  --output artifacts/eval/ftb/seed-20260729
```

Prepare real visual features with `fold2reason data visual --help`. Then evaluate each checkpoint:

```bash
torchrun --standalone --nproc_per_node=4 -m fold2reason evaluate spatial \
  --config configs/eval/spatial.toml \
  --model models/Qwen3.5-9B \
  --adapter artifacts/runs/full/seed-20260729/final/adapter \
  --model-name full-seed-20260729 \
  --spatialviz-root data/benchmarks/data/external/spatialviz_bench \
  --vsi-root data/benchmarks/data/external/vsi_bench \
  --visual-cache artifacts/cache/visual \
  --output artifacts/eval/spatial/seed-20260729
```

Run the spatial command separately for the base model, omitting `--adapter` and changing the output/model name. Text and FTB commands perform paired base/adapter evaluation. Base predictions may be reused only with identical model, sample IDs, prompt, generation, and scoring protocols. Never replace real image/video input with text-only substitutes.

## Aggregate three seeds

Normalize evaluator results into the schema shown in `examples/score_record.json`: one record per model/seed/benchmark, with `base` and `adapted` as fractions in `[0,1]`. The normalizer step is explicit because the backend output schemas differ; select the canonical accuracy/score field for each benchmark.

```bash
fold2reason analyze general10 \
  --input artifacts/eval/general10_scores.json \
  --output artifacts/eval/general10_summary.json
```

The aggregator requires exactly three complete seeds for each model, rejects duplicate records and unknown benchmarks, and reports mean scores, paired percentage-point changes, seed SD, and the equal-weight ten-benchmark macro. Validate sample coverage in the raw evaluator outputs before normalizing scores; the aggregator verifies score-table completeness, not raw prediction coverage.
