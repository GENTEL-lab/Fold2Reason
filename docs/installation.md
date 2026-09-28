# Installation

The validation environment has Python 3.12, torch 2.10.0, torchvision 0.25.0, accelerate 1.14.0, NumPy 2.5.1, pyarrow 25.0.0, safetensors 0.8.0, av 18.0.0, einops 0.8.2, and Pillow 12.3.0. These are observed local versions, not a claim that every combination allowed by the dependency ranges has been tested.

Transformer integration is pinned to:

| Package | Source revision |
|---|---|
| Transformers | `dff4572dfa4bfa9f00cc8414e4b84877552fefe9` |
| PEFT | `3116c96b5d870851e8477eb99dfcf8471bac0e72` |

Both revisions come from the working environment's installed package metadata. Git-based installation requires Git and network access. PyTorch and torchvision must be a matching CUDA pair. Install them for the target machine before installing this package.

```bash
python -m pip install -e '.[vision]'
fold2reason doctor
```

The `vision` extra is needed for SpatialViz/VSI preprocessing and evaluation. The `plots` extra adds Matplotlib. Optimized attention and convolution kernels are optional environment choices: no machine-specific binary wheel URL is hard-coded into the package.

Model loaders use local model directories. Obtain the original model through its publisher, retain the tokenizer/configuration files, and pass the directory with `--model`. Some non-Qwen compatibility loaders execute model-supplied remote code; review that code before using those optional paths.

Use four GPUs for the reference distributed recipes. GPU memory depends on sequence and media lengths; the research recipe was run on 80 GB accelerators. CPU tests exercise algorithms and contracts, not the memory/performance requirements of the full model.

