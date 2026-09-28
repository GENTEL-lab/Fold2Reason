# Release checklist

## Included in this candidate

- Installable `src/fold2reason` package with explicit package imports.
- Unified data, training, evaluation, and aggregation commands.
- Full-method, frozen-decoder, and FoldingCorpus-only training recipes.
- Full General-10 text/FTB/spatial recipes.
- Synthetic examples, unit tests, provenance, and a release-content scanner.
- Pinned model/dataset downloads, safe media extraction and full-suite input verification.
- Six data archives, one original frozen-decoder-heads archive and one Qwen3.5-9B main experiment LoRA/workspace archive selected for upload. All have checksums and an installer. Base-model weights are excluded.

## Validate before publication

- [ ] Choose and add the project LICENSE; maintainer confirmation is pending.
- [ ] Confirm source ownership and any required third-party attribution.
- [ ] Finalize the public paper citation, author list, repository URL, and support channel.
- [ ] Upload the prepared resource archives and replace the null URLs in `docs/release_assets.json`.
- [ ] Attach the original OpenFold distillation-corpus acquisition record and confirm its redistribution terms; the exact selected data are packaged but this provenance decision remains open.
- [ ] Run a fresh-install GPU smoke test, then a complete end-to-end reproduction on an independent machine.
- [ ] Review the final staged Git diff and run `python tools/check_release.py --strict`.

The [packaging validation report](validation.md) records checks actually run during this migration. It does not certify that model training or a full benchmark suite has been rerun. Historical cluster orchestration, exploratory variants, manuscript archives, and API experiments remain in the private research repository.
