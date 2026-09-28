"""Download pinned public model/benchmark resources and unpack their media."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


def safe_unzip(archive: Path, destination: Path) -> None:
    """Extract regular files only; reject traversal, symlinks, and overwrites."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError("Unsafe archive member path")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Archive symlinks are not supported")
            target = destination.joinpath(*path.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Archive member escapes destination")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                # Re-extraction validates existing files instead of overwriting them.
                import zlib
                crc = 0
                with target.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        crc = zlib.crc32(chunk, crc)
                if target.stat().st_size != member.file_size or crc != member.CRC:
                    raise FileExistsError(f"Existing file differs: {target.name}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extracting")
            with bundle.open(member) as source, temporary.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/resources.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--resource", default="general10-external",
                        help="model, general10-external (nine datasets), or an individual manifest key")
    parser.add_argument("--list", action="store_true", help="Show pinned selections without downloading")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()
    resources = json.loads(args.manifest.read_text())["resources"]
    selected = ([name for name, item in resources.items() if item["repo_type"] == "dataset"]
                if args.resource == "general10-external" else [args.resource])
    if set(selected) - resources.keys():
        parser.error("Unknown resource; inspect configs/resources.json")
    if args.list:
        print(json.dumps({name: resources[name] for name in selected}, indent=2))
        return
    from huggingface_hub import snapshot_download
    for name in selected:
        item = resources[name]
        destination = args.root / item["destination"]
        if not destination.resolve().is_relative_to(args.root.resolve()):
            raise ValueError("Resource destination escapes root")
        print(f"Downloading {name} at {item['revision']}", flush=True)
        snapshot_download(repo_id=item["repo_id"], repo_type=item["repo_type"],
                          revision=item["revision"], allow_patterns=item["allow_patterns"],
                          local_dir=destination)
        if not args.skip_extract:
            for archive in item.get("extract_zip", []):
                print(f"Extracting {name}/{archive}", flush=True)
                extract_to = destination / item.get("extract_destinations", {}).get(archive, ".")
                if not extract_to.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Extraction directory escapes resource destination")
                safe_unzip(destination / archive, extract_to)
        (destination / "SOURCE.json").write_text(json.dumps(item, indent=2) + "\n")
    print("Download complete. Run the input verifier before evaluation.")
