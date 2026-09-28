#!/usr/bin/env python3
"""Build a geometry-eval cache from FoldBench monomer_protein targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from fold2reason.data.backbone import to_bb4q10
from fold2reason.data.geometry_cache import (
    MARKER,
    SYSTEM,
    build_skeleton,
    geometry_targets,
    quantiles,
    tokenize_record,
)


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "C",
    "PYL": "K",
}
ATOM_ORDER = ("N", "CA", "C", "O")
FOLD_SFT_SYSTEM = (
    "You predict protein backbone structures from sequence, MSA and "
    "template-derived constraints. Output only a valid BB4Q10 structure "
    "block; do not add explanations."
)
FOLD_SFT_INSTRUCTION = (
    "Predict the query backbone. Return only BB4Q10 in the canonical "
    "PCA-v1 frame."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def length_bin(length: int) -> str:
    if length < 80:
        return "short"
    if length < 180:
        return "medium"
    if length < 350:
        return "long"
    return "xlong"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_loop(lines: list[str], start: int) -> tuple[list[str], list[list[str]], int]:
    tags = []
    rows = []
    index = start + 1
    while index < len(lines) and lines[index].startswith("_"):
        tags.append(lines[index].split()[0])
        index += 1
    current: list[str] = []
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped == "#":
            index += 1
            break
        if stripped == "loop_" or stripped.startswith("_"):
            break
        current.extend(shlex.split(stripped))
        while len(current) >= len(tags) and tags:
            rows.append(current[: len(tags)])
            current = current[len(tags) :]
        index += 1
    return tags, rows, index


def read_loop_tags_and_row_start(
    lines: list[str],
    start: int,
) -> tuple[list[str], int]:
    tags = []
    index = start + 1
    while index < len(lines) and lines[index].startswith("_"):
        tags.append(lines[index].split()[0])
        index += 1
    return tags, index


def skip_loop_rows(lines: list[str], start: int) -> int:
    index = start
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == "#":
            return index + 1
        if stripped == "loop_" or stripped.startswith("_"):
            return index
        index += 1
    return index


def extract_backbone_from_cif(
    cif_path: Path,
    chain_id: str,
    manifest_sequence: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    lines = cif_path.read_text(errors="replace").splitlines()
    index = 0
    atom_site_tags = None
    atom_site_rows = None
    while index < len(lines):
        if lines[index].strip() == "loop_":
            tags, row_start = read_loop_tags_and_row_start(lines, index)
            if (
                "_atom_site.Cartn_x" in tags
                and "_atom_site.label_asym_id" in tags
            ):
                atom_site_tags, atom_site_rows, _ = parse_loop(lines, index)
                break
            index = skip_loop_rows(lines, row_start)
        else:
            index += 1
    if atom_site_tags is None or atom_site_rows is None:
        raise ValueError(f"{cif_path}: _atom_site loop not found")
    col = {tag: i for i, tag in enumerate(atom_site_tags)}
    required = [
        "_atom_site.group_PDB",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_atom_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]
    missing = [tag for tag in required if tag not in col]
    if missing:
        raise ValueError(f"{cif_path}: missing atom_site columns {missing}")
    model_col = col.get("_atom_site.pdbx_PDB_model_num")
    alt_col = col.get("_atom_site.label_alt_id")
    residues: dict[int, dict[str, Any]] = {}
    for row in atom_site_rows:
        if row[col["_atom_site.group_PDB"]] != "ATOM":
            continue
        if model_col is not None and row[model_col] not in {"1", ".", "?"}:
            continue
        if row[col["_atom_site.label_asym_id"]] != chain_id:
            continue
        atom_name = row[col["_atom_site.label_atom_id"]].strip('"')
        if atom_name not in ATOM_ORDER:
            continue
        if alt_col is not None and row[alt_col] not in {".", "?", "A"}:
            continue
        seq_id_text = row[col["_atom_site.label_seq_id"]]
        if seq_id_text in {".", "?"}:
            continue
        seq_id = int(seq_id_text)
        residue = residues.setdefault(
            seq_id,
            {
                "resname": row[col["_atom_site.label_comp_id"]].upper(),
                "atoms": {},
            },
        )
        residue["atoms"].setdefault(
            atom_name,
            np.asarray(
                [
                    float(row[col["_atom_site.Cartn_x"]]),
                    float(row[col["_atom_site.Cartn_y"]]),
                    float(row[col["_atom_site.Cartn_z"]]),
                ],
                dtype=np.float32,
            ),
        )
    ordered = [residues[key] for key in sorted(residues)]
    parsed_sequence = "".join(
        AA3_TO_1.get(item["resname"], "X") for item in ordered
    )
    coords = np.zeros((len(ordered), 4, 3), dtype=np.float32)
    mask = np.zeros(len(ordered), dtype=bool)
    for residue_index, item in enumerate(ordered):
        atoms = item["atoms"]
        if all(atom in atoms for atom in ATOM_ORDER):
            coords[residue_index] = np.stack([atoms[atom] for atom in ATOM_ORDER])
            mask[residue_index] = True
    if int(mask.sum()) < max(3, round(0.5 * len(mask))):
        raise ValueError(f"{cif_path}: insufficient complete backbone residues")
    if len(parsed_sequence) == len(manifest_sequence):
        return coords, mask, parsed_sequence
    compact_coords = []
    compact_mask = []
    for item in ordered:
        atoms = item["atoms"]
        if all(atom in atoms for atom in ATOM_ORDER):
            compact_coords.append(np.stack([atoms[atom] for atom in ATOM_ORDER]))
            compact_mask.append(True)
        else:
            compact_coords.append(np.zeros((4, 3), dtype=np.float32))
            compact_mask.append(False)
    compact_coords_array = np.asarray(compact_coords, dtype=np.float32)
    compact_mask_array = np.asarray(compact_mask, dtype=bool)
    if int(compact_mask_array.sum()) < max(
        3, round(0.5 * len(compact_mask_array))
    ):
        raise ValueError(f"{cif_path}: insufficient compact backbone residues")
    return compact_coords_array, compact_mask_array, parsed_sequence


def read_a3m_records(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    records = []
    name = None
    chunks: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks)))
            name = line[1:].strip().split()[0] or f"row{len(records)}"
            chunks = []
        elif name is not None:
            chunks.append(line.strip())
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def a3m_to_query_aligned(sequence: str, aligned: str) -> str:
    kept = []
    for char in aligned:
        if char.islower() or char == ".":
            continue
        if char == "-":
            kept.append("-")
        elif char.isalpha():
            kept.append(char.upper())
    if len(kept) < len(sequence):
        kept.extend(["-"] * (len(sequence) - len(kept)))
    return "".join(kept[: len(sequence)])


def msa_block(
    entry: dict[str, Any],
    root: Path,
    sequence: str,
    max_rows: int,
) -> tuple[str, int]:
    records = read_a3m_records(root / "msa" / f"{entry['id']}.a3m")
    if not records:
        return "", 0
    lines = [
        (
            f'<msa selected_rows="{min(len(records), max_rows)}" '
            f'source_depth="{len(records)}" selection="foldbench_a3m_order">'
        )
    ]
    for index, (name, aligned) in enumerate(records[:max_rows]):
        label = "Q" if index == 0 else f"H{index}"
        aligned_query = a3m_to_query_aligned(sequence, aligned)
        lines.append(f"{label} {aligned_query} DEL=-")
    lines.append("</msa>")
    return "\n".join(lines), min(len(records), max_rows)


def templates_block(entry: dict[str, Any], root: Path) -> tuple[str, int]:
    path = root / "templates" / f"{entry['id']}.json"
    if not path.exists():
        return "", 0
    payload = json.loads(path.read_text())
    templates = payload.get("templates") or []
    lines = ['<templates representation="foldbench_template_metadata">']
    for template in templates[:4]:
        name = (
            template.get("name")
            or template.get("id")
            or template.get("pdb_id")
            or "template"
        )
        coverage = template.get("coverage", template.get("query_coverage", "?"))
        release = template.get("release_date", template.get("release", "?"))
        lines.append(f"{name} release={release} coverage={coverage}")
    lines.append("</templates>")
    return "\n".join(lines), len(templates)


def make_user_content(
    entry: dict[str, Any],
    root: Path,
    sequence: str,
    max_msa_rows: int,
) -> tuple[str, str, int, int]:
    parts = [
        '<protein_folding_input version="foldbench-monomer-v1">',
        f'<sequence length="{len(sequence)}">',
        sequence,
        "</sequence>",
    ]
    msa, msa_rows = msa_block(entry, root, sequence, max_msa_rows)
    templates, template_count = templates_block(entry, root)
    if msa:
        parts.append(msa)
    if templates:
        parts.append(templates)
    parts.extend([FOLD_SFT_INSTRUCTION, "</protein_folding_input>"])
    if msa and template_count:
        view = "SEQ_MSA_TPL"
    elif msa:
        view = "SEQ_MSA"
    else:
        view = "SEQ"
    return "\n".join(parts), view, msa_rows, template_count


def make_record(
    entry: dict[str, Any],
    root: Path,
    max_msa_rows: int,
) -> dict[str, Any]:
    cif_path = root / "ground_truths" / f"{entry['assembly_name']}.cif"
    coords, mask, parsed_sequence = extract_backbone_from_cif(
        cif_path, entry["chain"], entry["seq"]
    )
    user_content, view, msa_rows, template_count = make_user_content(
        entry, root, parsed_sequence, max_msa_rows
    )
    return {
        "id": f"foldbench:{entry['id']}",
        "messages": [
            {"role": "system", "content": FOLD_SFT_SYSTEM},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": to_bb4q10(parsed_sequence, coords)},
        ],
        "metadata": {
            "source_id": entry["id"],
            "pdb": entry["pdb"],
            "chain": entry["chain"],
            "assembly_name": entry["assembly_name"],
            "source_dataset": "FoldBench/monomer_protein",
            "sequence_length": len(parsed_sequence),
            "length_bin": length_bin(len(parsed_sequence)),
            "foldbench_manifest_sequence": entry["seq"],
            "foldbench_manifest_length": len(entry["seq"]),
            "cif_parsed_sequence": parsed_sequence,
            "cif_parsed_length": len(parsed_sequence),
            "input_view": view,
            "msa_rows": msa_rows,
            "template_count": template_count,
            "coordinate_residue_loss_mask": mask.tolist(),
            "coordinate_residue_loss_weight": mask.astype(np.float32).tolist(),
        },
    }


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
        "--foldbench-root",
        type=Path,
        default=Path("data/foldbench/monomer_protein"),
    )
    parser.add_argument(
        "--sft-output",
        type=Path,
        default=project / "foldbench_monomer_qwen35_sft.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "artifacts/cache" / "qwen35_foldbench_monomer_geometry.pt",
    )
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-msa-rows", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=0)
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
    entries = read_jsonl(args.foldbench_root / "monomer_protein.jsonl")
    if args.max_examples:
        entries = entries[: args.max_examples]
    records = []
    skipped = []
    for index, entry in enumerate(entries, 1):
        try:
            records.append(make_record(entry, args.foldbench_root, args.max_msa_rows))
        except Exception as exc:
            skipped.append({"id": entry.get("id"), "error": str(exc)})
        if index % 50 == 0 or index == len(entries):
            print(
                f"[foldbench-cache] converted {index}/{len(entries)} "
                f"kept={len(records)} skipped={len(skipped)}",
                flush=True,
            )
    args.sft_output.parent.mkdir(parents=True, exist_ok=True)
    with args.sft_output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tokenized = []
    token_skipped = []
    for record in records:
        try:
            tokenized.append(
                tokenize_record(tokenizer, marker_id, record, args.max_length)
            )
        except Exception as exc:
            token_skipped.append({"id": record["id"], "error": str(exc)})
    splits = {"validation": tokenized}
    stats = {
        "status": "PASS" if tokenized else "FAIL",
        "format_version": 2,
        "dataset": "FoldBench/monomer_protein",
        "model": str(args.model),
        "marker": MARKER,
        "marker_token_id": marker_id,
        "max_length": args.max_length,
        "max_msa_rows": args.max_msa_rows,
        "foldbench_root": str(args.foldbench_root),
        "sft_output": str(args.sft_output),
        "sources": {
            "monomer_protein": {
                "path": str(args.foldbench_root / "monomer_protein.jsonl"),
                "sha256": sha256(args.foldbench_root / "monomer_protein.jsonl"),
            },
        },
        "conversion": {
            "input_entries": len(entries),
            "records": len(records),
            "skipped": skipped,
            "tokenized": len(tokenized),
            "token_skipped": token_skipped,
        },
        "splits": {name: summarize(rows) for name, rows in splits.items()},
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
    args.output.with_suffix(".stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(json.dumps(stats, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
