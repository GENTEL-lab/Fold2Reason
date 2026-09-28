# Reproduction workflow

Run commands from the repository root. Keep models in `models/`, externally obtained datasets in `data/`, and generated caches/checkpoints in `artifacts/`. All three directories are ignored by Git. Use a fresh output directory for every run.

First follow [resource acquisition](resources.md). Installing the core training, FoldBench334 and frozen-decoder-heads archives supplies the cached inputs and original decoder used below; skip sections 1–3 and start training at section 4. To skip training entirely, install the main experiment's three-seed LoRA/workspace archive and follow [pretrained evaluation](evaluation.md#evaluate-the-released-main-checkpoints). Base-model weights are downloaded separately. Download URLs for owner-hosted archives will be filled after upload. The preparation commands remain available for rebuilding from raw records or training a different decoder.

## 1. Prepare geometry tensors

The training input is JSONL in the research BB4Q10 chat-record format: a system message, a user message with sequence/MSA/template evidence, and an assistant backbone target. See [data formats](data.md). The core resource archive provides the selected 1,000/100/100 records and matching training caches. Rebuilding a different source collection requires a separately documented selection pipeline.

```bash
fold2reason data geometry \
  --model models/Qwen3.5-9B \
  --data-root data/protein_training \
  --train-file data/protein_training/train.jsonl \
  --validation-file data/protein_training/dev.jsonl \
  --validation-rare-file data/protein_training/test.jsonl \
  --output artifacts/cache/geometry.pt

fold2reason data foldbench \
  --model models/Qwen3.5-9B \
  --foldbench-root data/foldbench/monomer_protein \
  --sft-output artifacts/cache/foldbench_targets.jsonl \
  --output artifacts/cache/foldbench334.pt
```

Keep training/dev/test protein IDs disjoint and exclude benchmark proteins from training. The FoldingCorpus builder checks exact-sequence overlap and 5-mer similarity against the supplied FoldBench cache. This audit does not establish homology-level independence; retain your source dataset's split and clustering manifest.

## 2. Build FoldingCorpus and the workspace cache

```bash
fold2reason data corpus \
  --model models/Qwen3.5-9B \
  --geometry-cache artifacts/cache/geometry.pt \
  --foldbench-cache artifacts/cache/foldbench334.pt \
  --balanced-labels \
  --output-dir artifacts/corpus \
  --output-cache artifacts/cache/corpus_tokens.pt

fold2reason data workspace \
  --input-cache artifacts/cache/corpus_tokens.pt \
  --negative-index artifacts/corpus/hard_negative_index.json \
  --output-cache artifacts/cache/foldingcorpus.pt \
  --manifest artifacts/corpus/workspace_manifest.json
```

The corpus packs twelve question/answer labels per protein, including the historical 32-way topology question. The workspace cache builder checks candidate counts and split provenance. A tiny example is insufficient for the 32-candidate protocol; use adequately sized, split-isolated candidate pools. The full recipe disables the **independent retrieval loss/forward**; this is distinct from the topology question in FoldingCorpus.

## 3. Train and freeze a geometry decoder

```bash
torchrun --standalone --nproc_per_node=4 -m fold2reason train decoder \
  --config configs/train/decoder.toml \
  --model models/Qwen3.5-9B \
  --cache artifacts/cache/geometry.pt \
  --output artifacts/decoder
```

The decoder recipe freezes the language model and the zero-initialized LoRA adapter and trains the geometry heads. Use the chosen decoder checkpoint consistently across comparison arms. Its files, seed, selection rule, and hash belong in the experiment record. For the original experimental decoder, install `fold2reason-frozen-decoder-heads-v1.tar.gz` instead of retraining it with this example recipe.

## 4. Train the full model and the Pure-LoRA baseline

```bash
torchrun --standalone --nproc_per_node=4 -m fold2reason train full \
  --config configs/train/full.toml \
  --model models/Qwen3.5-9B \
  --cache artifacts/cache/foldingcorpus.pt \
  --frozen-decoder-checkpoint artifacts/decoder/final \
  --output artifacts/runs/full/seed-20260729

torchrun --standalone --nproc_per_node=4 -m fold2reason train corpus-only \
  --config configs/train/corpus_only.toml \
  --model models/Qwen3.5-9B \
  --cache artifacts/cache/foldingcorpus.pt \
  --output artifacts/runs/corpus-only/seed-20260729
```

Repeat with seeds `20260803` and `20260804`. The full model updates LoRA and workspace parameters while holding the decoder fixed. The Pure-LoRA baseline updates the language adapter only. The full recipe disables training-time benchmark evaluation; evaluate the declared terminal or checkpoint states offline.

For scaling, prepare nested protein sets before training and retain the per-set manifests. The reference fixed-epoch series uses 50, 100, 250, 500, 1,000, 2,000, and 4,000 proteins. Matching an epoch count alone does not ensure identical data quality, token exposure, or sample ordering.

## 5. Evaluate

Follow [evaluation](evaluation.md). Structural metrics are measured with the frozen decoder. General-10 uses the adapted language model without workspace input. Record the exact base model, adapter, decoder, prompts, generation settings, sample IDs, and evaluator version.
