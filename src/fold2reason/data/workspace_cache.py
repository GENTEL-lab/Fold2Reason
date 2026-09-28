#!/usr/bin/env python3
"""Add deterministic topology fingerprints and 32-way teacher banks to Phase-1 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


GRID_SIZE = 32
FINGERPRINT_DIM = 64
PROJECTION_SEED = 20260804


def parse_args() -> argparse.Namespace:
    project = Path.cwd()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-cache",
        type=Path,
        default=project / "artifacts/cache/openfold_spatial_bridge_v2_2_independent.pt",
    )
    parser.add_argument(
        "--negative-index",
        type=Path,
        default=project / "data/openfold_spatial_bridge_v2_2_independent/hard_negative_index.json",
    )
    parser.add_argument(
        "--output-cache",
        type=Path,
        default=project / "artifacts/cache/openfold_phase2_workspace_v0.pt",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project / "data/openfold_phase2_workspace_v0/manifest.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def build_projection() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PROJECTION_SEED)
    feature_dim = 3 * GRID_SIZE * GRID_SIZE
    random_matrix = torch.randn(feature_dim, FINGERPRINT_DIM, generator=generator)
    projection, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return projection.float()


def topology_features(row: dict[str, Any]) -> torch.Tensor:
    coords = row["target_coords"].float()[:, 1]
    mask = row["residue_mask"].bool()
    coords = coords[mask]
    distances = torch.cdist(coords, coords)
    distance_kernel = torch.exp(-distances / 10.0)
    contacts = (distances < 8.0).float()
    clipped_distance = (distances / 30.0).clamp(max=1.0)
    channels = torch.stack([distance_kernel, contacts, clipped_distance]).unsqueeze(0)
    resized = F.interpolate(
        channels,
        size=(GRID_SIZE, GRID_SIZE),
        mode="bilinear",
        align_corners=True,
    )[0]
    return resized.reshape(-1)


def main() -> None:
    args = parse_args()
    cache = torch.load(args.input_cache, map_location="cpu", weights_only=False)
    negative_index = json.loads(args.negative_index.read_text())
    projection = build_projection()

    row_by_id: dict[str, dict[str, Any]] = {}
    split_by_id: dict[str, str] = {}
    fingerprints: dict[str, torch.Tensor] = {}
    for split, rows in cache["splits"].items():
        for row in rows:
            row_by_id[row["id"]] = row
            split_by_id[row["id"]] = split
            fingerprints[row["id"]] = F.normalize(
                topology_features(row) @ projection,
                dim=0,
            )

    split_name = {"train": "train", "validation": "dev", "validation_rare": "frozen_test"}
    copied_splits: dict[str, list[dict[str, Any]]] = {}
    oracle_correct = 0
    checked = 0
    distractor_split_violations = []
    candidate_count_violations = []
    for source_split, rows in cache["splits"].items():
        index = negative_index["splits"][split_name[source_split]]
        copied_rows = []
        for row in rows:
            negative_ids = index[row["id"]]["negative_ids"]
            candidate_ids = [row["id"], *negative_ids]
            if len(candidate_ids) != 32:
                candidate_count_violations.append(row["id"])
            if source_split != "train":
                bad = [value for value in negative_ids if split_by_id[value] != "train"]
                if bad:
                    distractor_split_violations.append({"id": row["id"], "bad": bad})
            bank = torch.stack([fingerprints[value] for value in candidate_ids])
            scores = bank @ fingerprints[row["id"]]
            oracle_correct += int(int(scores.argmax()) == 0)
            checked += 1
            copied = dict(row)
            copied["workspace"] = {
                "teacher_fingerprint": fingerprints[row["id"]].to(torch.float16),
                "candidate_fingerprints": bank.to(torch.float16),
                "candidate_ids": candidate_ids,
                "matched_donor_id": negative_ids[0],
                "exact_negative_count": index[row["id"]]["exact_negative_count"],
            }
            copied_rows.append(copied)
        copied_splits[source_split] = copied_rows

    augmented = {
        "format_version": 2,
        "stats": {
            **cache["stats"],
            "workspace_version": "phase2_workspace_v0",
            "teacher_grid_size": GRID_SIZE,
            "teacher_fingerprint_dim": FINGERPRINT_DIM,
            "teacher_projection_seed": PROJECTION_SEED,
            "teacher_projection_sha256": tensor_sha256(projection),
        },
        "teacher_projection": projection,
        "splits": copied_splits,
    }
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(augmented, args.output_cache)

    qc = {
        "rows_checked": checked,
        "candidate_count_violations": candidate_count_violations,
        "heldout_distractor_split_violations": distractor_split_violations,
        "oracle_teacher_retrieval_r_at_1": oracle_correct / max(checked, 1),
        "factual_source_cache": str(args.input_cache),
        "source_cache_sha256": sha256(args.input_cache),
        "negative_index_sha256": sha256(args.negative_index),
        "output_cache_sha256": sha256(args.output_cache),
        "teacher_projection_sha256": tensor_sha256(projection),
    }
    status = (
        "COMPLETE"
        if not candidate_count_violations
        and not distractor_split_violations
        and qc["oracle_teacher_retrieval_r_at_1"] == 1.0
        else "FAILED_QC"
    )
    manifest = {
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_version": "phase2_workspace_v0",
        "grid_size": GRID_SIZE,
        "fingerprint_dim": FINGERPRINT_DIM,
        "projection_seed": PROJECTION_SEED,
        "splits": {name: len(rows) for name, rows in copied_splits.items()},
        "qc": qc,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if status != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
