#!/usr/bin/env python3
"""Check the publication tree without printing matched credential values.

This is a heuristic local check, not a replacement for staged-diff review or a
dedicated secret scanner. Run before adding external artifacts to the repository.
"""
import argparse
from pathlib import Path
import re
import sys

PATTERNS = {
    "Hugging Face credential": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "API credential": re.compile(r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "machine-specific user path": re.compile(r"(?<![\w/.-])/(?:data\d*|\d+data|home)/[A-Za-z][\w.-]+/"),
    "private network address": re.compile(r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"),
}
EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist"}
EXTERNAL_ROOTS = {"artifacts", "data", "models", "outputs", "logs"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".pem", ".key", ".parquet", ".zip", ".tar", ".gz"}


def scan(root):
    errors, warnings = [], []
    scanned = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if set(relative.parts) & EXCLUDED or any(part.endswith(".egg-info") for part in relative.parts):
            continue
        # Local models/data are ignored by Git. Flag them if actually tracked.
        if relative.parts[0] in EXTERNAL_ROOTS:
            continue
        if path.is_symlink():
            errors.append(f"{relative}: symbolic link requires explicit publication review")
            continue
        if not path.is_file():
            continue
        scanned += 1
        if path.name in {".env", "id_rsa", "id_ed25519", ".netrc"} or path.suffix in FORBIDDEN_SUFFIXES:
            errors.append(f"{relative}: sensitive or unreviewed binary artifact")
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            errors.append(f"{relative}: file larger than 5 MiB requires artifact review")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            errors.append(f"{relative}: non-text resource requires publication review")
            continue
        for number, line in enumerate(content.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    errors.append(f"{relative}:{number}: {label}")
    # Check even force-added files in ignored local-output directories.
    if (root / ".git").exists():
        import subprocess
        result = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
        for name in result.stdout.decode().split("\0"):
            if name and Path(name).parts[0] in EXTERNAL_ROOTS:
                errors.append(f"{name}: local data/model/output must not be tracked")
        for name in EXTERNAL_ROOTS:
            if (root / name).exists():
                ignored = subprocess.run(["git", "check-ignore", "-q", name + "/"], cwd=root)
                if ignored.returncode != 0:
                    errors.append(f"{name}/: local resource directory must be excluded by .gitignore")
    if not any((root / name).is_file() for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")):
        warnings.append("Project LICENSE is pending maintainer confirmation.")
    return scanned, errors, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--strict", action="store_true", help="Treat pending release decisions as failures")
    args = parser.parse_args()
    scanned, errors, warnings = scan(args.root.resolve())
    print(f"Scanned {scanned} publication files.")
    for message in errors:
        print("ERROR: " + message)
    for message in warnings:
        print("WARNING: " + message)
    print(f"{len(errors)} errors; {len(warnings)} pending release decisions.")
    return int(bool(errors or (args.strict and warnings)))


if __name__ == "__main__":
    sys.exit(main())
