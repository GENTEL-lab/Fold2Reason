#!/usr/bin/env python3
"""Build the leakage-safe residue-marker cache for the 1k-v2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer


MARKER = "<|object_ref_start|>"
SYSTEM = (
    "Predict the protein backbone from the provided sequence and optional "
    "evolutionary or template evidence. Associate each residue marker with "
    "its N, CA, C, and O coordinates."
)
OLD_INSTRUCTION = (
    "Predict the query backbone. Return only BB4Q10 in the canonical "
    "PCA-v1 frame."
)
NEW_INSTRUCTION = (
    "Predict the query backbone for each residue marker in the assistant "
    "backbone query."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: list[float]) -> dict[str, float]:
    return {
        str(q): round(float(np.quantile(values, q)), 4)
        for q in (0.0, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0)
    }


def parse_sequence(user_content: str) -> str:
    match = re.search(
        r'<sequence length="(\d+)">\n([A-Z]+)\n</sequence>',
        user_content,
    )
    if not match:
        raise ValueError("Sequence block not found")
    sequence = match.group(2)
    if len(sequence) != int(match.group(1)):
        raise ValueError("Sequence length attribute mismatch")
    return sequence


def parse_backbone(content: str, sequence: str) -> np.ndarray:
    lines = content.splitlines()
    if not lines or not lines[0].startswith(
        '<structure format="BB4Q10"'
    ):
        raise ValueError("Invalid BB4Q10 header")
    if lines[-1] != "</structure>":
        raise ValueError("Invalid BB4Q10 closing tag")
    rows = []
    observed = []
    for expected_index, line in enumerate(lines[1:-1], 1):
        fields = line.split()
        if len(fields) != 14 or int(fields[0]) != expected_index:
            raise ValueError(f"Invalid BB4Q10 row {expected_index}")
        observed.append(fields[1])
        rows.append(
            np.asarray(
                [int(value) for value in fields[2:]],
                dtype=np.float32,
            ).reshape(4, 3)
            / 10.0
        )
    if "".join(observed) != sequence:
        raise ValueError("BB4Q10 amino-acid sequence mismatch")
    return np.stack(rows)


def normalize_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return vector / np.maximum(
        np.linalg.norm(vector, axis=-1, keepdims=True), eps
    )


def dihedral_sincos(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1_unit = normalize_vector(b1)
    v = b0 - np.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - np.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit
    v = normalize_vector(v)
    w = normalize_vector(w)
    cosine = np.sum(v * w, axis=-1)
    sine = np.sum(np.cross(b1_unit, v) * w, axis=-1)
    return np.stack([sine, cosine], axis=-1).astype(np.float32)


def backbone_torsions(
    coords: np.ndarray,
    residue_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    length = len(coords)
    torsions = np.zeros((length, 3, 2), dtype=np.float32)
    mask = np.zeros((length, 3), dtype=bool)
    if length < 2:
        return torsions, mask
    # phi_i = C_(i-1), N_i, CA_i, C_i
    torsions[1:, 0] = dihedral_sincos(
        coords[:-1, 2],
        coords[1:, 0],
        coords[1:, 1],
        coords[1:, 2],
    )
    mask[1:, 0] = residue_mask[:-1] & residue_mask[1:]
    # psi_i = N_i, CA_i, C_i, N_(i+1)
    torsions[:-1, 1] = dihedral_sincos(
        coords[:-1, 0],
        coords[:-1, 1],
        coords[:-1, 2],
        coords[1:, 0],
    )
    mask[:-1, 1] = residue_mask[:-1] & residue_mask[1:]
    # omega_i = CA_i, C_i, N_(i+1), CA_(i+1)
    torsions[:-1, 2] = dihedral_sincos(
        coords[:-1, 1],
        coords[:-1, 2],
        coords[1:, 0],
        coords[1:, 1],
    )
    mask[:-1, 2] = residue_mask[:-1] & residue_mask[1:]
    return torsions, mask


def geometry_targets(
    coords: np.ndarray,
    residue_mask: np.ndarray,
) -> dict[str, torch.Tensor | float | int]:
    ca = coords[:, 1]
    distances = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    indices = np.arange(len(coords))
    separation = np.abs(indices[:, None] - indices[None, :])
    pair_mask = residue_mask[:, None] & residue_mask[None, :]
    contacts = (distances < 8.0) & (separation >= 6) & pair_mask
    valid_ca = ca[residue_mask]
    center = valid_ca.mean(axis=0)
    rg = float(
        np.sqrt(np.mean(np.sum((valid_ca - center) ** 2, axis=-1)))
    )
    torsions, torsion_mask = backbone_torsions(coords, residue_mask)
    return {
        "ca_distances": torch.tensor(distances, dtype=torch.float16),
        "contacts": torch.tensor(contacts, dtype=torch.bool),
        "torsion_sincos": torch.tensor(torsions, dtype=torch.float32),
        "torsion_mask": torch.tensor(torsion_mask, dtype=torch.bool),
        "radius_of_gyration": rg,
        "long_range_contacts": int(np.triu(contacts, k=1).sum()),
    }


def build_skeleton(sequence: str) -> str:
    lines = [
        '<backbone_query atoms="N,CA,C,O" marker="RES3D">',
        *[
            f"{index} {amino_acid} {MARKER}"
            for index, amino_acid in enumerate(sequence, 1)
        ],
        "</backbone_query>",
    ]
    return "\n".join(lines)


def infer_input_view(user_content: str) -> str:
    has_msa = "<msa " in user_content
    has_templates = (
        "<templates " in user_content
        or "<template_contacts " in user_content
    )
    if has_msa and has_templates:
        return "SEQ_MSA_TPL"
    if has_msa:
        return "SEQ_MSA"
    return "SEQ"


def tokenize_record(
    tokenizer: Any,
    marker_id: int,
    record: dict[str, Any],
    max_length: int,
) -> dict[str, Any]:
    user_content = record["messages"][1]["content"]
    sequence = parse_sequence(user_content)
    metadata = record["metadata"]
    if len(sequence) != int(metadata["sequence_length"]):
        raise ValueError(f"{record['id']}: metadata sequence mismatch")
    user_content = user_content.replace(OLD_INSTRUCTION, NEW_INSTRUCTION)
    if OLD_INSTRUCTION in user_content:
        raise ValueError(f"{record['id']}: old output instruction remained")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": build_skeleton(sequence)},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    input_ids = tokenizer(
        rendered,
        add_special_tokens=False,
    )["input_ids"]
    marker_positions = [
        index for index, token_id in enumerate(input_ids) if token_id == marker_id
    ]
    if len(marker_positions) != len(sequence):
        raise ValueError(
            f"{record['id']}: {len(marker_positions)} markers for "
            f"{len(sequence)} residues"
        )
    if len(input_ids) > max_length:
        raise ValueError(
            f"{record['id']}: {len(input_ids)} > max_length {max_length}"
        )

    coords = parse_backbone(record["messages"][2]["content"], sequence)
    if "coordinate_residue_loss_mask" in metadata:
        residue_mask = np.asarray(
            metadata["coordinate_residue_loss_mask"], dtype=bool
        )
        residue_weights = np.asarray(
            metadata["coordinate_residue_loss_weight"], dtype=np.float32
        )
    else:
        residue_mask = np.ones(len(sequence), dtype=bool)
        residue_weights = np.ones(len(sequence), dtype=np.float32)
    if not (
        len(residue_mask) == len(residue_weights) == len(sequence)
    ):
        raise ValueError(f"{record['id']}: geometry weight length mismatch")
    if residue_mask.sum() < max(3, round(0.5 * len(sequence))):
        raise ValueError(f"{record['id']}: insufficient valid residues")
    targets = geometry_targets(coords, residue_mask)

    return {
        "id": record["id"],
        "source_id": metadata["source_id"],
        "sequence": sequence,
        "input_view": metadata.get(
            "input_view", infer_input_view(user_content)
        ),
        "length_bin": metadata["length_bin"],
        "sequence_length": len(sequence),
        "input_ids": torch.tensor(input_ids, dtype=torch.int32),
        "marker_positions": torch.tensor(
            marker_positions, dtype=torch.int32
        ),
        "target_coords": torch.tensor(coords, dtype=torch.float32),
        "residue_mask": torch.tensor(residue_mask, dtype=torch.bool),
        "residue_weights": torch.tensor(
            residue_weights, dtype=torch.float16
        ),
        **targets,
    }


def tokenize_split(
    tokenizer: Any,
    marker_id: int,
    path: Path,
    max_length: int,
) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for index, line in enumerate(handle, 1):
            if not line.strip():
                continue
            rows.append(
                tokenize_record(
                    tokenizer,
                    marker_id,
                    json.loads(line),
                    max_length,
                )
            )
            if index % 100 == 0:
                print(f"[geometry-cache:{path.stem}] {index}", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = [len(row["input_ids"]) for row in rows]
    residues = [row["sequence_length"] for row in rows]
    valid_fraction = [
        float(row["residue_mask"].float().mean()) for row in rows
    ]
    contacts = [row["long_range_contacts"] for row in rows]
    rg = [row["radius_of_gyration"] for row in rows]
    by_view: dict[str, list[int]] = defaultdict(list)
    for row, tokens in zip(rows, input_tokens):
        by_view[row["input_view"]].append(tokens)
    return {
        "records": len(rows),
        "total_input_tokens": int(sum(input_tokens)),
        "input_token_quantiles": quantiles(input_tokens),
        "residue_quantiles": quantiles(residues),
        "valid_residue_fraction_quantiles": quantiles(valid_fraction),
        "long_range_contact_quantiles": quantiles(contacts),
        "radius_of_gyration_quantiles": quantiles(rg),
        "views": dict(Counter(row["input_view"] for row in rows)),
        "by_view_input_token_quantiles": {
            view: quantiles(values)
            for view, values in sorted(by_view.items())
        },
    }


def parse_args() -> argparse.Namespace:
    project = Path.cwd()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3.5-9B"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project
        / "openfold_highconf_1k_qwen35_lora_pilot_v1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project
        / "artifacts/cache"
        / "qwen35_openfold_1k_v2_geometry.pt",
    )
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--validation-rare-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, use_fast=True
    )
    marker_ids = tokenizer.encode(MARKER, add_special_tokens=False)
    if len(marker_ids) != 1:
        raise ValueError(f"{MARKER} is not one token: {marker_ids}")
    marker_id = marker_ids[0]
    split_paths = {
        "train": args.train_file
        or args.data_root / "train_mixed_1000.jsonl",
        "validation": args.validation_file
        or args.data_root / "validation_mixed_100.jsonl",
    }
    validation_rare = args.validation_rare_file or (
        args.data_root / "validation_rare_mixed_100.jsonl"
    )
    if validation_rare.is_file():
        split_paths["validation_rare"] = validation_rare
    splits = {
        name: tokenize_split(
            tokenizer, marker_id, path, args.max_length
        )
        for name, path in split_paths.items()
    }
    stats = {
        "status": "PASS",
        "format_version": 2,
        "model": str(args.model),
        "marker": MARKER,
        "marker_token_id": marker_id,
        "max_length": args.max_length,
        "system_prompt": SYSTEM,
        "sources": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in split_paths.items()
        },
        "splits": {
            name: summarize(rows) for name, rows in splits.items()
        },
    }
    payload = {
        "format_version": 2,
        "stats": stats,
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    stats_path = args.output.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
