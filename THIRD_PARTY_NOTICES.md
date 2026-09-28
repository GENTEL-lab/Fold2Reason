# Third-party resources

This source-release candidate depends on PyTorch, Transformers, PEFT, Accelerate, NumPy, PyArrow, safetensors, einops, Pillow, tqdm, and optional torchvision, PyAV, and Matplotlib. They are installed as dependencies and retain their upstream licenses. Their source or binaries are not vendored in this repository.

The release contains reorganized project implementation files, deterministic synthetic examples, and configuration/documentation written for this package. The provenance inventory identifies migrated files. Maintainers must confirm ownership and attribution before selecting the project license.

Base-model weights, protein data, learned checkpoints, benchmark questions/media, and proprietary fonts are excluded from the Git/source package. The separate upload archives contain selected project data, the processed FoldBench cache, the original frozen decoder and the Qwen3.5-9B main experiment's three-seed LoRA/workspace weights. Base-model weights are excluded. Archive notices retain Qwen/FoldBench attributions and identify pending project-license and distillation-data provenance decisions. See [resource acquisition](docs/resources.md). Nine external General-10 datasets are downloaded directly from their pinned publishers.
