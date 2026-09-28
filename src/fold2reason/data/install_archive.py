"""Verify and install a project resource archive into a checkout."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def install(archive, root, expected_sha256):
    with archive.open("rb") as handle:
        observed = hashlib.file_digest(handle, "sha256").hexdigest()
    if observed != expected_sha256:
        raise ValueError("Archive SHA-256 mismatch")
    root = root.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        seen = set()
        for member in members:
            relative = PurePosixPath(member.name)
            if (not member.isfile() or relative.is_absolute() or ".." in relative.parts
                    or "\\" in member.name or not relative.parts
                    or relative.parts[0] not in {"data", "artifacts"}):
                raise ValueError("Unsafe resource archive member")
            target = root.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError("Resource member escapes checkout")
            if target.exists() or target.is_symlink() or target in seen:
                raise FileExistsError(f"Refusing to replace existing or duplicate resource: {relative}")
            seen.add(target)
        for member in members:
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    return len(members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("docs/release_assets.json"))
    args = parser.parse_args()
    catalog = json.loads(args.manifest.read_text())
    entries = {item["filename"]: item for item in catalog["archives"]}
    if args.archive.name not in entries:
        parser.error("Archive is not listed in the release manifest")
    count = install(args.archive, args.root, entries[args.archive.name]["sha256"])
    print(f"Verified and installed {count} files from {args.archive.name}")
