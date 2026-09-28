# Fold2Reason

Protein-structure post-training for language models. Fold2Reason combines **FoldingCorpus** supervision with a shared spatial workspace and a frozen geometry decoder. General reasoning is evaluated with the adapted language model; structural readout uses the workspace and decoder.

This repository contains the Qwen3.5 training and evaluation pipeline, a FoldingCorpus-only Pure-LoRA baseline, deterministic data preparation, and General-10 score aggregation.

Start here: [install](#install) → [connect `dataset.zip`](#project-bundle-datasetzip) → [download the base model and external benchmarks](#models-and-evaluation-data) → [evaluate released checkpoints](#evaluate-the-released-checkpoints-no-training) or [retrain](#train).

## Install

Use Python 3.12 and a CUDA-compatible PyTorch installation. The tested environment uses PyTorch 2.10.0. Transformers and PEFT are pinned to the source revisions recorded in this repository; see [installation](docs/installation.md).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[vision]'
fold2reason doctor
fold2reason --help
```

Install the matching PyTorch/torchvision CUDA wheels for your system before the editable install. Run all commands below from the repository root in this environment. The source checkout contains code and recipes; project data and trained weights are distributed separately in `dataset.zip`.

## Project bundle: `dataset.zip`

Download [dataset.zip from Google Drive](https://drive.google.com/file/d/1QfcZrQc1Bg16KBOWo40HM8t2bPDtty7I/view?usp=sharing) and save it as `data/dataset.zip`. For command-line download, use [gdown](https://github.com/wkentaro/gdown):

```bash
python -m pip install gdown
mkdir -p data
gdown 1QfcZrQc1Bg16KBOWo40HM8t2bPDtty7I \
  --output data/dataset.zip --continue
python -m zipfile --test data/dataset.zip
```

`--continue` resumes an interrupted download. If Google Drive temporarily limits automated downloads, use the browser link and save the ZIP at the same path. After the ZIP integrity check passes, follow [Extract and connect the bundle](#extract-and-connect-the-bundle).

The archive contains the main experiment's data, frozen decoder, and three trained LoRA/workspace checkpoints:

```text
dataset/
├── fold2reason-frozen-decoder-heads/
│   ├── geometry_config.json
│   └── geometry_heads.pt
├── fold2reason-ftb-core-v1/
│   ├── train.jsonl.gz
│   ├── validation.jsonl.gz
│   ├── test.jsonl.gz
│   └── test_ood.jsonl.gz
├── fold2reason-qwen35-9b-main-lora-workspace/
│   ├── pretrained/qwen35-9b-main/
│   │   ├── seed-20260729/
│   │   ├── seed-20260803/
│   │   └── seed-20260804/
│   │       # Each seed directory contains:
│   │       # adapter/adapter_config.json, adapter/adapter_model.safetensors,
│   │       # workspace.pt, workspace_config.json
│   └── resource_manifests/
│       └── fold2reason-qwen35-9b-main-lora-workspace-3seeds-v1/
│           ├── LICENSE-Qwen.txt
│           ├── README.md
│           └── checkpoint_manifest.json
├── fold2reason-training-core-v1/
│   ├── corpus/
│   │   ├── train.jsonl
│   │   ├── dev.jsonl
│   │   ├── frozen_test.jsonl
│   │   ├── hard_negative_index.json
│   │   ├── independence_audit.json
│   │   ├── recomputation_audit.json
│   │   ├── manifest.json
│   │   └── split_hashes.json
│   └── protein_training/
│       ├── train.jsonl
│       ├── dev.jsonl
│       └── test.jsonl
└── foldbench334/
    ├── target_manifest.json
    └── targets_bb4q10.jsonl
```

The 34 files cover 1,000 training / 100 development / 100 held-out proteins, their FoldingCorpus supervision, all four FTB-Core splits, the processed 334-protein FoldBench snapshot, and the main experiment's fixed epoch-3 / step-375 checkpoints. Keep each adapter paired with its same-seed workspace. The frozen decoder is shared across seeds.

The official Qwen3.5-9B weights and nine external General-10 datasets are downloaded separately below. Tokenized training/FoldBench caches, visual-feature caches, and the 50–4,000-protein scaling collections are **not included** in this ZIP. Cache-building instructions follow; the bundle supports the 1,000-protein main experiment.

### Extract and connect the bundle

Run this once after downloading. It accepts archives with or without an outer `dataset/` directory, preserves the uploaded layout, and creates directory symlinks at the paths used by the recipes. Re-running checks existing links; conflicting destinations raise an error instead of being replaced.

```bash
python - <<'PY'
from pathlib import Path
from fold2reason.data.download import safe_unzip

unpacked = Path("data/project_bundle")
safe_unzip(Path("data/dataset.zip"), unpacked)
root = unpacked / "dataset" if (unpacked / "dataset").is_dir() else unpacked
weights = root / "fold2reason-qwen35-9b-main-lora-workspace"
links = {
    "data/protein_training": root / "fold2reason-training-core-v1/protein_training",
    "artifacts/corpus": root / "fold2reason-training-core-v1/corpus",
    "artifacts/foldbench334": root / "foldbench334",
    "data/benchmarks/data/generated/ftb_core/v1": root / "fold2reason-ftb-core-v1",
    "artifacts/decoder/final": root / "fold2reason-frozen-decoder-heads",
    "artifacts/pretrained/qwen35-9b-main": weights / "pretrained/qwen35-9b-main",
    "artifacts/resource_manifests/fold2reason-qwen35-9b-main-lora-workspace-3seeds-v1":
        weights / "resource_manifests/fold2reason-qwen35-9b-main-lora-workspace-3seeds-v1",
}
for destination, source in links.items():
    target = Path(destination)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if (target.exists() or target.is_symlink()) and target.resolve() != source.resolve():
        raise FileExistsError(f"Destination already in use: {target}")
for destination, source in links.items():
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_symlink():
        target.symlink_to(source.resolve(), target_is_directory=True)
print("Bundle ready:", root.resolve())
PY
```

FTB-Core is read directly from `.jsonl.gz`; leave those four files compressed. The source directories remain under `data/project_bundle/`, so keep that directory while using the symlinks. The older `data install-archive` command and `docs/release_assets.json` describe separate `.tar.gz` packages, not this ZIP; use the extraction procedure above for this release.

## Models and evaluation data

The starting model is [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B). Here “base” means the checkpoint before Fold2Reason training; use this exact model, not a similarly named `Base` or quantized variant. The pinned revision is `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; its four weight shards, model index, configuration and tokenizer match the local experimental files by SHA-256.

```bash
# Run from this checkout after installation.
fold2reason data download --resource model
# Download the nine external General-10 datasets at the recorded revisions.
# Includes SpatialViz images and VSI videos, and unpacks their archives.
fold2reason data download --resource general10-external
```

The downloader installs each resource in the directory expected by the evaluators. To inspect sources without downloading, append `--list`; to fetch one dataset, use its key from [configs/resources.json](configs/resources.json), for example `--resource graphqa_easy`. The equivalent base-model command is:

```bash
hf download Qwen/Qwen3.5-9B \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --local-dir models/Qwen3.5-9B
```

| Evaluation set | Download source | Full evaluation rows |
|---|---|---:|
| FTB-Core | [`dataset.zip`](#project-bundle-datasetzip): `fold2reason-ftb-core-v1/` | 12,000 |
| SpatialViz | [SpatialViz-Bench](https://huggingface.co/datasets/Anonymous285714/SpatialViz-Bench) | 1,180 |
| VSI | [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) | 5,130 |
| GraphQA Easy | [GraphQA_Easy](https://huggingface.co/datasets/tonysun9/GraphQA_Easy) | 21,600 |
| GraphQA Hard | [GraphQA_Hard](https://huggingface.co/datasets/tonysun9/GraphQA_Hard) | 21,600 |
| BBH | [BIG-Bench Hard snapshot](https://huggingface.co/datasets/lukaemon/bbh) | 6,511 |
| ChemBench | [ChemBench](https://huggingface.co/datasets/jablonkagroup/ChemBench) | 2,148 |
| ChemBench4K | [ChemBench4K](https://huggingface.co/datasets/AI4Chem/ChemBench4K) | 4,009 |
| Lab-Bench | [LAB-Bench](https://huggingface.co/datasets/futurehouse/lab-bench) | 1,967 |
| SciBench | [SciBench](https://huggingface.co/datasets/xw27/scibench) | 580 |
| FoldBench structural readout | [Official FoldBench](https://github.com/BEAM-Labs/FoldBench); use the processed `foldbench334/` snapshot in [`dataset.zip`](#project-bundle-datasetzip) | 334 proteins |

General-10 uses **76,725 examples** across the first ten rows. Counts describe this evaluator's complete usable inputs, not the sizes of every split in the upstream repositories. FTB-Core uses `test` + `test_ood` (6,000 each); Lab-Bench excludes image-dependent questions in the text evaluator. The pinned external downloads are recorded in [configs/resources.json](configs/resources.json). See [resource acquisition](docs/resources.md#public-downloads) for external file formats and publisher terms; this README defines the current single-ZIP installation workflow.

After connecting the bundle and downloading the external datasets:

```bash
fold2reason data verify --benchmark-root data/benchmarks
```

This checks all ten counts, unique example IDs and real image/video paths without running model inference.

## Evaluate the released checkpoints (no training)

General-10 uses **Qwen3.5-9B + LoRA**. Structural readout additionally uses the **same seed's workspace + frozen decoder**. Tokenizer files come from the official base-model download. The following commands use four GPUs; repeat the evaluation with `20260803` and `20260804` for the other two seeds.

### Full General-10

Build the real image/video feature cache once, then reuse it across seeds and base-model evaluation:

```bash
torchrun --standalone --nproc_per_node=4 -m fold2reason data visual \
  --model models/Qwen3.5-9B \
  --spatialviz-root data/benchmarks/data/external/spatialviz_bench \
  --vsi-root data/benchmarks/data/external/vsi_bench \
  --num-video-frames 32 \
  --output artifacts/cache/visual
```

Run all three evaluators to cover the ten benchmarks. These recipes use full evaluation data, with no mini selection manifest:

```bash
EVAL_SEED=20260729
MAIN_CKPT="artifacts/pretrained/qwen35-9b-main/seed-${EVAL_SEED}"

torchrun --standalone --nproc_per_node=4 -m fold2reason evaluate general10-text \
  --config configs/eval/general10_text.toml \
  --model models/Qwen3.5-9B --adapter "${MAIN_CKPT}/adapter" \
  --general-benchs-root data/benchmarks \
  --output "artifacts/eval/text/seed-${EVAL_SEED}"

torchrun --standalone --nproc_per_node=4 -m fold2reason evaluate ftb \
  --config configs/eval/ftb.toml \
  --model models/Qwen3.5-9B --adapter "${MAIN_CKPT}/adapter" \
  --benchmark-root data/benchmarks/data/generated/ftb_core/v1 \
  --output "artifacts/eval/ftb/seed-${EVAL_SEED}"

torchrun --standalone --nproc_per_node=4 -m fold2reason evaluate spatial \
  --config configs/eval/spatial.toml \
  --model models/Qwen3.5-9B --adapter "${MAIN_CKPT}/adapter" \
  --model-name "full-seed-${EVAL_SEED}" \
  --spatialviz-root data/benchmarks/data/external/spatialviz_bench \
  --vsi-root data/benchmarks/data/external/vsi_bench \
  --visual-cache artifacts/cache/visual \
  --output "artifacts/eval/spatial/seed-${EVAL_SEED}"
```

`general10-text` covers seven text benchmarks; `ftb` covers FTB-Core; `spatial` covers SpatialViz and VSI. Text and FTB commands evaluate both base and adapter. Run the spatial base once separately, with identical feature and generation settings:

```bash
torchrun --standalone --nproc_per_node=4 -m fold2reason evaluate spatial \
  --config configs/eval/spatial.toml \
  --model models/Qwen3.5-9B --model-name base \
  --spatialviz-root data/benchmarks/data/external/spatialviz_bench \
  --vsi-root data/benchmarks/data/external/vsi_bench \
  --visual-cache artifacts/cache/visual \
  --output artifacts/eval/spatial/base
```

After all three seeds finish, follow [score normalization and aggregation](docs/evaluation.md#aggregate-three-seeds) to report mean scores, paired gains and seed SD. The equal-weight General-10 macro averages ten benchmark scores, not all examples pooled together.

### FoldBench: all 334 proteins

The ZIP supplies rendered BB4Q10 records, not `foldbench334.pt`. Build the validation cache from these records using the downloaded tokenizer. This CPU step uses the included sequence/MSA/template text and coordinate targets; no upstream structure or MSA download is needed.

```bash
python - <<'PY'
from pathlib import Path
import torch
from transformers import AutoTokenizer
from fold2reason.data.geometry_cache import MARKER, sha256, summarize, tokenize_split

source = Path("artifacts/foldbench334/targets_bb4q10.jsonl")
output = Path("artifacts/cache/foldbench334.pt")
if output.exists():
    raise FileExistsError(f"Cache already exists: {output}")
tokenizer = AutoTokenizer.from_pretrained("models/Qwen3.5-9B", local_files_only=True)
marker = tokenizer.encode(MARKER, add_special_tokens=False)
assert len(marker) == 1, marker
rows = tokenize_split(tokenizer, marker[0], source, 32768)
assert len(rows) == len({row["id"] for row in rows}) == 334
payload = {"format_version": 2, "splits": {"validation": rows},
           "stats": {"source_sha256": sha256(source), "validation": summarize(rows)}}
output.parent.mkdir(parents=True, exist_ok=True)
torch.save(payload, output)
print("Saved", output, "with", len(rows), "targets")
PY

torchrun --standalone --nproc_per_node=4 -m fold2reason evaluate foldbench \
  --model models/Qwen3.5-9B \
  --checkpoint artifacts/pretrained/qwen35-9b-main/seed-20260729 \
  --frozen-decoder-checkpoint artifacts/decoder/final \
  --cache artifacts/cache/foldbench334.pt --max-examples 0 \
  --label full-seed-20260729 \
  --output artifacts/eval/foldbench/full-seed-20260729
```

Repeat the evaluation command for the other seeds; reuse the same cache and decoder. The evaluator reports lDDT-Cα, contact F1 and the repository's alignment-based TM-score readout. It consumes the processed 334-target snapshot, not an arbitrary current upstream FoldBench selection. Further metric details are in [structural evaluation](docs/evaluation.md#structural-readout).

## Train

Use this route to retrain the main 1,000-protein experiment. The ZIP already provides its original frozen decoder, so decoder training is unnecessary. First build `artifacts/cache/geometry.pt` and `artifacts/cache/foldingcorpus.pt` using the preparation steps below; the `.jsonl` files cannot be passed directly to `--cache`.

<details>
<summary>Prepare training caches from the ZIP (CPU; run once before training)</summary>

First build the 334-target cache in the preceding section; it is used for the corpus overlap audit, not as training data. Then tokenize the released training splits:

```bash
fold2reason data geometry \
  --model models/Qwen3.5-9B \
  --data-root data/protein_training \
  --train-file data/protein_training/train.jsonl \
  --validation-file data/protein_training/dev.jsonl \
  --validation-rare-file data/protein_training/test.jsonl \
  --output artifacts/cache/geometry.pt

fold2reason data corpus \
  --model models/Qwen3.5-9B \
  --geometry-cache artifacts/cache/geometry.pt \
  --foldbench-cache artifacts/cache/foldbench334.pt \
  --balanced-labels --seed 20260804 \
  --output-dir artifacts/corpus_rebuilt \
  --output-cache artifacts/cache/corpus_tokens.pt
```

The corpus command regenerates supervision as well as packing tokens. Keep its output separate from the released `artifacts/corpus/` and verify that the regenerated records and negative index match before training. The preparation seed `20260804` is fixed for all three training seeds.

```bash
python - <<'PY'
import json
from pathlib import Path

released, rebuilt = Path("artifacts/corpus"), Path("artifacts/corpus_rebuilt")
for name, count in [("train", 12000), ("dev", 1200), ("frozen_test", 1200)]:
    def records(root):
        return [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines() if line.strip()]
    original, regenerated = records(released), records(rebuilt)
    assert len(original) == count, (name, len(original))
    assert original == regenerated, f"Released/rebuilt corpus mismatch: {name}"
name = "hard_negative_index.json"
assert json.loads((released / name).read_text()) == json.loads((rebuilt / name).read_text())
print("Released FoldingCorpus records and negative index match")
PY

fold2reason data workspace \
  --input-cache artifacts/cache/corpus_tokens.pt \
  --negative-index artifacts/corpus/hard_negative_index.json \
  --output-cache artifacts/cache/foldingcorpus.pt \
  --manifest artifacts/cache/workspace_manifest.json
```

Continue only after the comparison passes and the workspace builder reports `COMPLETE`. The final cache contains 1,000 / 100 / 100 proteins with twelve labels each. Reuse it across training seeds. The released corpus audits and split hashes remain unchanged; rebuilt artifacts have their own provenance and hashes.

The CPU rebuild was checked against the reference caches: all geometry inputs/targets and FoldingCorpus token/label tensors matched exactly, as did the 14,400 released records and negative index. Auxiliary retrieval fingerprints can differ slightly with the CPU linear-algebra backend; the main recipe disables that retrieval forward pass and loss. Rebuilt `.pt` files therefore need not have the original cache's file hash.

</details>

Train on four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 -m fold2reason train full \
  --config configs/train/full.toml \
  --model models/Qwen3.5-9B \
  --cache artifacts/cache/foldingcorpus.pt \
  --frozen-decoder-checkpoint artifacts/decoder/final \
  --output artifacts/runs/full/seed-20260729
```

The same command with `--dry-run` prints the resolved backend arguments without loading a model. CLI arguments override recipe values. Use `--seed 20260803` and `--seed 20260804` with separate output directories for the remaining seeds.

For the Pure-LoRA comparison, use `train corpus-only` with `configs/train/corpus_only.toml` and the same cache; see [training details](docs/reproduction.md#4-train-the-full-model-and-the-pure-lora-baseline). The longer reproduction guide also describes the older cache-containing archives and optional decoder training. For this `dataset.zip`, follow the preparation above. Input schemas are described in [data formats](docs/data.md).

## Layout

```text
Fold2Reason/
├── src/fold2reason/
│   ├── data/          # FoldingCorpus, geometry and visual caches
│   ├── models/        # Language backbone, workspace and geometry heads
│   ├── losses/        # Structural objectives
│   ├── training/      # Frozen decoder, full method and Pure LoRA
│   ├── evaluation/    # Structural readout, text, image/video and FTB
│   └── analysis/      # Paired statistics and General-10 aggregation
├── configs/           # Small, editable training/evaluation recipes
├── examples/          # Synthetic data and score schemas
├── docs/              # Setup, reproduction, data and release notes
├── tests/             # CPU unit tests and packaging contracts
└── tools/             # Repository-wide release checks
```

## Tests

```bash
python -m unittest discover -s tests -v
python tools/check_release.py
```

Tests run on CPU. See the [validation report](docs/validation.md) for completed checks. Full GPU training and dataset-wide evaluation are separate reproduction steps.

## Release status

This is a source-release candidate. The [release checklist](docs/release_checklist.md) records validation and remaining publication decisions, including the project license and resource redistribution permissions. The current `dataset.zip` is linked above. Its archive SHA-256 is pending verification; the older per-archive hashes do not apply to the repacked ZIP. The ZIP integrity check verifies the archive's contents against its internal CRC values, not against a publisher-provided checksum.

The public name is **FoldingCorpus**. Legacy tensor keys, checkpoint fields, and some internal class names retain `relation` for compatibility with existing checkpoints. Public commands accept `--loss-foldingcorpus` and `--corpus-variant`.

## Attribution

The implementation was reorganized from the research codebase. [Provenance](docs/provenance.json) records the source-to-package mapping and source hashes. Dependencies retain their own licenses; see [third-party resources](THIRD_PARTY_NOTICES.md). Citation metadata can be added when the paper's public title, author list, and identifier are finalized.
