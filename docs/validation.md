# Migration validation

Validated on 2026-09-14; resource-acquisition checks added the same day. This report separates package checks from experiment reproduction.

| Check | Result |
|---|---|
| CPU unit suite | 31 tests passed |
| Public command backends | All 18 command help entries loaded without models or datasets |
| Training/evaluation recipes | All seven TOML recipes accepted by their backend parsers |
| Training objective smoke test | Selected-answer CE matches the causal token shift; ignored positions receive no logit gradient |
| Workspace smoke test | Expected tensor shapes, normalized attention, and finite nonzero gradients |
| Structural metric smoke test | Kabsch alignment and lDDT are invariant to a rigid transform on synthetic coordinates |
| Synthetic FoldingCorpus examples | Both displayed contact answers match executable labeling rules |
| General-10 aggregation | Three-seed means/SD checked; missing, duplicate, unknown, and invalid scores rejected |
| Python syntax | Package, tests, and tools compile successfully |
| Migration provenance | All 23 migrated module hashes match the recorded snapshot |
| Source/wheel build | Both distribution formats built successfully |
| Wheel import and CLI | Installed without dependencies into a separate temporary environment; import, help, and doctor work outside the checkout |
| Publication-content scanner | No flagged credentials, private paths, network addresses, or unreviewed binary resources in the publication tree |

CPU backend tests used the existing research dependencies listed in [installation](installation.md), with the new package on the import path. The separate wheel check did not install tensor dependencies: it verifies packaging and the lightweight CLI, not a fresh installation of the complete GPU stack.

The strict release check deliberately fails while the project LICENSE is pending. No new training, model inference, dataset-wide prediction/scoring run, or remote publication was performed. A fresh-install GPU smoke test and independent end-to-end reproduction remain on the [release checklist](release_checklist.md).

## Resource validation

- Verified the nine pinned external dataset revisions through the Hugging Face API; all requested source file layouts exist.
- Matched the four local Qwen3.5-9B weight-shard SHA-256 hashes to upstream LFS metadata, and matched its tokenizer, tokenizer configuration, model index and model configuration to revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Read all ten local evaluation inputs with the actual loaders: **76,725 unique-within-benchmark rows**, with the expected per-benchmark counts and existing real image/video paths. This is input validation, not model evaluation.
- Downloaded the pinned SciBench snapshot into a temporary directory using the new downloader; the loader reads 580 usable questions.
- Extracted the actual SpatialViz archive into a fresh temporary directory and verified that the loader resolves all 1,180 image questions. Inspected all three VSI zip member paths; they use the expected video-directory prefixes. VSI videos were not downloaded again.
- Built and read back eight archives; verified each member's SHA-256, and verified tensor-payload equality after cleaning private provenance paths from caches/checkpoints.
- The owner subsequently selected a data-only release: six data archives remain in the upload directory; the decoder and pretrained-checkpoint archives were moved to local-only storage without deletion. The public manifest and upload checksums list only the six data archives.
- The release then added the original decoder at the owner's request: six data archives plus a smaller heads-only decoder archive, with no adapter/tokenizer files. Earlier folders remain intact.
- The current `upload_with_decoder/` folder additionally includes **Qwen3.5-9B main-experiment LoRA/workspace weights only**, for seeds 20260729, 20260803 and 20260804 at epoch 3 / step 375. Its manifest now lists eight archives. The main-result mapping was checked against the paper's result aggregation code and each run's completion/configuration records. LoRA files are byte-identical to the selected training checkpoints; all workspace tensors match the actual full-evaluation runtime copies. Their only runtime/source difference was decoder-path metadata.
- Installed the new main-weight archive and heads-only decoder into a temporary checkout. All three PEFT configs parse, workspace state dictionaries load strictly, and the workspace/frozen coordinate decoder produce finite `(12, 4, 3)` coordinates on synthetic hidden states in CPU smoke tests. This does not constitute a full language-model or GPU inference run.
- Installed the actual FTB-Core archive into a temporary checkout using the public hash-verifying installer. Unit tests also exercise invalid hashes, traversal, symlinks, and overwrite rejection.

The resource release still needs owner-hosted URLs, the project's license, and the original distillation-training-data acquisition/redistribution record. Downloading related OpenProteinSet data is not a substitute for establishing that record.

## Repeat local checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests tools
python tools/check_release.py
```

For a publication build, install the `dev` extra and run `python -m build`. The source archive includes recipes, documentation, examples, tests, and the release checker. The wheel contains the Python package; use a checkout or source archive for the recipes and reproduction guide.
