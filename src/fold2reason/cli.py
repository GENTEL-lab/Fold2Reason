"""A lightweight CLI; tensor libraries are imported only by selected commands."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tomllib

COMMANDS = {
    ("data", "download"): "data.download",
    ("data", "verify"): "data.verify_resources",
    ("data", "install-archive"): "data.install_archive",
    ("data", "geometry"): "data.geometry_cache",
    ("data", "foldbench"): "data.foldbench_cache",
    ("data", "corpus"): "data.folding_corpus",
    ("data", "workspace"): "data.workspace_cache",
    ("data", "visual"): "data.visual_cache",
    ("train", "decoder"): "training.geometry",
    ("train", "full"): "training.workspace",
    ("train", "corpus-only"): "training.pure_lora",
    ("evaluate", "corpus"): "evaluation.folding_corpus",
    ("evaluate", "geometry"): "evaluation.geometry",
    ("evaluate", "foldbench"): "evaluation.foldbench",
    ("evaluate", "general10-text"): "evaluation.general_text",
    ("evaluate", "ftb"): "evaluation.ftb",
    ("evaluate", "spatial"): "evaluation.spatial",
    ("analyze", "general10"): "analysis.general10",
}
ALIASES = {"--loss-foldingcorpus": "--loss-relation", "--corpus-variant": "--relation-variant"}


def config_arguments(path: Path) -> list[str]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if set(data) - {"description", "arguments"}:
        raise ValueError("Config supports only description and [arguments].")
    arguments = []
    for key, value in data.get("arguments", {}).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                arguments.append(flag)
            continue
        values = value if isinstance(value, list) else [value]
        arguments.append(flag)
        for item in values:
            rendered = os.path.expandvars(str(item))
            if "${" in rendered:
                raise ValueError(f"Unresolved environment variable for {flag}")
            arguments.append(rendered)
    return arguments


def doctor() -> None:
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "pyarrow", "av", "torchvision"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    print(json.dumps({"python": sys.version.split()[0], "packages": packages,
                      "note": "Reports installed versions; does not allocate GPU memory or download models."}, indent=2))


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv == ["--help"] or argv == ["-h"]:
        print("Fold2Reason\n\nUsage: fold2reason GROUP COMMAND [--config recipe.toml] [--dry-run] [OPTIONS]\n")
        for group, command in COMMANDS:
            print(f"  {group:9s} {command}")
        print("\n  doctor    Show installed versions\n\nUse GROUP COMMAND --help for backend options.")
        return
    if argv == ["doctor"]:
        doctor()
        return
    key = tuple(argv[:2])
    if key not in COMMANDS:
        raise SystemExit("Unknown command. Run fold2reason --help.")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    options, rest = parser.parse_known_args(argv[2:])
    inherited = config_arguments(options.config) if options.config else []
    args = [ALIASES.get(value, value) for value in inherited + rest]
    module_name = "fold2reason." + COMMANDS[key]
    if options.dry_run:
        print(json.dumps({"module": module_name, "arguments": args}, indent=2))
        return
    module = importlib.import_module(module_name)
    previous = sys.argv
    try:
        sys.argv = [module_name, *args]
        module.main()
    finally:
        sys.argv = previous
