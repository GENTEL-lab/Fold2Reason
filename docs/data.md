# Data and resource contracts

## External inputs

| Input | Expected local form | Included? |
|---|---|---|
| Base model | Model, tokenizer and config files in a Hugging Face-compatible directory | No |
| Training protein records | Disjoint train/dev/test BB4Q10 chat JSONL | No |
| FoldBench | `monomer_protein` directory with manifest/targets and available MSA/templates | No |
| General-10 datasets | Dataset-specific files consumed by `evaluation.general_text`, `evaluation.ftb`, and `evaluation.multimodal` | No |
| Frozen decoder and adapters | Checkpoint directories from the documented training stages | Generated separately |
| Synthetic example | `examples/toy_protein.json` | Yes |

Use the [pinned downloader and project resource archives](resources.md) to populate these inputs. Dataset licenses and model licenses are separate from the project's license. The downloader preserves the layout expected by the evaluators and extracts real image/video media; project archives supply the exact caches and sample selections.

## Geometry cache

`data.geometry_cache` parses the BB4Q10 assistant target and replaces the coordinate target in the language-model input with one marker per residue. Targets remain separate tensors. Rows include `id`, `sequence`, `input_ids`, `marker_positions`, `target_coords`, `residue_mask`, and auxiliary geometry targets. The cache uses `splits` with `train`, `validation`, and `validation_rare`.

Input target syntax and tokenization are defined by `parse_backbone`, `build_skeleton`, and `tokenize_record`. Treat them as the executable schema; `fold2reason data geometry --help` lists the file overrides. The tokenizer must recognize the configured residue marker as one token.

## FoldingCorpus

`data.folding_corpus` deterministically derives contact, distance ordering, orientation, center-distance, local-frame, chirality, multi-constraint, and topology-identification labels. It saves question arguments and evidence so labels can be recomputed from the source coordinates.

The training cache keeps the established `bridge`, `relation_*`, and related checkpoint keys for compatibility. They denote FoldingCorpus supervision in the public terminology. Renaming these serialized keys would invalidate existing caches/checkpoints, so public naming is separated from the storage schema.

The toy example is hand-constructed and identifies itself as synthetic. It illustrates a contact-label rule and is not a protein from a training or evaluation dataset. It is not a full tokenizer-dependent training cache.

## Trust boundary

Training caches and checkpoints use PyTorch serialization, including `torch.load(..., weights_only=False)` for structured caches. Load only files you created or obtained from a trusted source. Model code requiring `trust_remote_code=True` must be reviewed separately. Authentication belongs in your local environment or Hugging Face credential store, never in committed configuration files.
