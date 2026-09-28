#!/usr/bin/env python3
"""Deterministic BB4Q10/PDB export and geometry-cache quality control."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from fold2reason.data.geometry_cache import (
    backbone_torsions,
    geometry_targets,
    parse_backbone,
)


ATOM_NAMES = ("N", "CA", "C", "O")
THREE_LETTER = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


def to_bb4q10(sequence: str, coords: np.ndarray) -> str:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape != (len(sequence), 4, 3):
        raise ValueError(
            f"Expected {(len(sequence), 4, 3)}, got {coords.shape}"
        )
    quantized = np.rint(coords * 10.0).astype(np.int64)
    lines = [
        (
            f'<structure format="BB4Q10" residues="{len(sequence)}" '
            'unit="0.1_angstrom" frame="pca_v1" '
            'columns="i aa Nxyz CAxyz Cxyz Oxyz">'
        )
    ]
    for index, (amino_acid, atoms) in enumerate(
        zip(sequence, quantized), 1
    ):
        values = " ".join(str(value) for value in atoms.reshape(-1))
        lines.append(f"{index} {amino_acid} {values}")
    lines.append("</structure>")
    return "\n".join(lines)


def to_backbone_pdb(sequence: str, coords: np.ndarray) -> str:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape != (len(sequence), 4, 3):
        raise ValueError(
            f"Expected {(len(sequence), 4, 3)}, got {coords.shape}"
        )
    lines = []
    serial = 1
    for residue_index, (amino_acid, atoms) in enumerate(
        zip(sequence, coords), 1
    ):
        residue_name = THREE_LETTER.get(amino_acid, "UNK")
        for atom_name, xyz in zip(ATOM_NAMES, atoms):
            element = atom_name[0]
            lines.append(
                f"ATOM  {serial:5d} {atom_name:^4s} {residue_name:>3s} "
                f"A{residue_index:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}"
                f"{xyz[2]:8.3f}  1.00  0.00          {element:>2s}"
            )
            serial += 1
    lines.extend(["TER", "END"])
    return "\n".join(lines) + "\n"


def max_abs_or_zero(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def qc_row(row: dict) -> dict[str, float | int | str]:
    sequence = row["sequence"]
    coords = row["target_coords"].float().numpy()
    mask = row["residue_mask"].numpy()
    rebuilt = parse_backbone(to_bb4q10(sequence, coords), sequence)
    targets = geometry_targets(coords, mask)
    rebuilt_torsions, rebuilt_torsion_mask = backbone_torsions(coords, mask)
    torsion_mask = row["torsion_mask"].numpy()
    torsion_difference = (
        rebuilt_torsions[torsion_mask]
        - row["torsion_sincos"].numpy()[torsion_mask]
    )
    return {
        "id": row["id"],
        "residues": len(sequence),
        "bb4q10_max_abs_error_angstrom": max_abs_or_zero(
            rebuilt - coords
        ),
        "ca_distance_max_abs_error_angstrom": max_abs_or_zero(
            targets["ca_distances"].float().numpy()
            - row["ca_distances"].float().numpy()
        ),
        "contact_mismatch_count": int(
            np.count_nonzero(
                targets["contacts"].numpy() != row["contacts"].numpy()
            )
        ),
        "torsion_max_abs_error": max_abs_or_zero(torsion_difference),
        "torsion_mask_mismatch_count": int(
            np.count_nonzero(rebuilt_torsion_mask != torsion_mask)
        ),
        "radius_of_gyration_abs_error": abs(
            float(targets["radius_of_gyration"])
            - float(row["radius_of_gyration"])
        ),
        "pdb_atom_records": to_backbone_pdb(sequence, coords).count(
            "\nATOM"
        )
        + 1,
    }


def parse_args() -> argparse.Namespace:
    project = Path.cwd()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=project
        / "artifacts/cache"
        / "qwen35_openfold_1k_v2_geometry.pt",
    )
    parser.add_argument(
        "--qc-output",
        type=Path,
        default=project
        / "artifacts/cache"
        / "qwen35_openfold_1k_v2_geometry.qc.json",
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--row-id")
    parser.add_argument("--bb4q10-output", type=Path)
    parser.add_argument("--pdb-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    all_rows = [
        row for rows in cache["splits"].values() for row in rows
    ]
    if args.row_id:
        matches = [row for row in all_rows if row["id"] == args.row_id]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one row for {args.row_id}, got {len(matches)}"
            )
        selected = matches
    else:
        selected = random.Random(args.seed).sample(
            all_rows, min(args.samples, len(all_rows))
        )
    results = [qc_row(row) for row in selected]
    maxima = {
        "bb4q10_max_abs_error_angstrom": max(
            item["bb4q10_max_abs_error_angstrom"] for item in results
        ),
        "ca_distance_max_abs_error_angstrom": max(
            item["ca_distance_max_abs_error_angstrom"] for item in results
        ),
        "contact_mismatch_count": max(
            item["contact_mismatch_count"] for item in results
        ),
        "torsion_max_abs_error": max(
            item["torsion_max_abs_error"] for item in results
        ),
        "torsion_mask_mismatch_count": max(
            item["torsion_mask_mismatch_count"] for item in results
        ),
        "radius_of_gyration_abs_error": max(
            item["radius_of_gyration_abs_error"] for item in results
        ),
    }
    passed = (
        maxima["bb4q10_max_abs_error_angstrom"] < 1e-5
        and maxima["ca_distance_max_abs_error_angstrom"] < 1e-3
        and maxima["contact_mismatch_count"] == 0
        and maxima["torsion_max_abs_error"] < 1e-6
        and maxima["torsion_mask_mismatch_count"] == 0
        and maxima["radius_of_gyration_abs_error"] < 1e-6
    )
    payload = {
        "status": "PASS" if passed else "FAIL",
        "cache": str(args.cache),
        "seed": args.seed,
        "samples": len(results),
        "maxima": maxima,
        "rows": results,
    }
    args.qc_output.parent.mkdir(parents=True, exist_ok=True)
    args.qc_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    if args.bb4q10_output:
        row = selected[0]
        args.bb4q10_output.write_text(
            to_bb4q10(row["sequence"], row["target_coords"].numpy())
            + "\n"
        )
    if args.pdb_output:
        row = selected[0]
        args.pdb_output.write_text(
            to_backbone_pdb(row["sequence"], row["target_coords"].numpy())
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
