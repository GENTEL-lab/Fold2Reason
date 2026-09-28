#!/usr/bin/env python3
"""Build deterministic Phase-1 protein relation data and an augmented train cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer


GENERATOR_VERSION = "openfold_spatial_bridge_v2.0"
BALANCED_GENERATOR_VERSION = "openfold_spatial_bridge_v2.2-independent"
RETRIEVAL_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef")
SPLIT_NAMES = {"train": "train", "validation": "dev", "validation_rare": "frozen_test"}


def parse_args() -> argparse.Namespace:
    project = Path.cwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-cache", type=Path, default=project / "artifacts/cache/qwen35_openfold_1k_v2_geometry.pt")
    parser.add_argument("--foldbench-cache", type=Path, default=project / "artifacts/cache/qwen35_foldbench_monomer_geometry.pt")
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3.5-9B"))
    parser.add_argument("--output-dir", type=Path, default=project / "data/openfold_spatial_bridge_v2")
    parser.add_argument("--output-cache", type=Path, default=project / "artifacts/cache/openfold_spatial_bridge_v2.pt")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--balanced-labels",
        action="store_true",
        help="Construct each operator with an approximately uniform answer distribution.",
    )
    return parser.parse_args()


def stable_seed(*items: object) -> int:
    digest = hashlib.sha256("::".join(map(str, items)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def angle_from_sincos(value: torch.Tensor) -> np.ndarray:
    array = value.float().numpy()
    return np.arctan2(array[..., 0], array[..., 1])


def structure_features(row: dict[str, Any]) -> dict[str, Any]:
    ca = row["target_coords"].float().numpy()[:, 1]
    mask = row["residue_mask"].numpy().astype(bool)
    valid = np.flatnonzero(mask)
    selected = ca[valid]
    centered = selected - selected.mean(0, keepdims=True)
    rg = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
    covariance = centered.T @ centered / max(len(centered), 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1].clip(1e-6)
    shape = [float(eigenvalues[1] / eigenvalues[0]), float(eigenvalues[2] / eigenvalues[0])]

    distances = np.linalg.norm(ca[:, None] - ca[None, :], axis=-1)
    separation = np.abs(np.arange(len(ca))[:, None] - np.arange(len(ca))[None, :])
    valid_pair = np.triu(mask[:, None] & mask[None, :] & (separation >= 12), 1)
    contacts = valid_pair & (distances < 8.0)
    pair_count = int(valid_pair.sum())
    contact_count = int(contacts.sum())
    density = contact_count / max(pair_count, 1)
    degrees = contacts.sum(0) + contacts.sum(1)
    degree_mean = float(degrees[mask].mean()) if mask.any() else 0.0
    span_mean = float(separation[contacts].mean() / max(len(ca), 1)) if contact_count else 0.0

    torsion = angle_from_sincos(row["torsion_sincos"])
    phi, psi = torsion[:, 0], torsion[:, 1]
    helix = (phi >= math.radians(-160)) & (phi <= math.radians(-20)) & (psi >= math.radians(-90)) & (psi <= math.radians(45))
    strand = (phi <= math.radians(-40)) & ((psi >= math.radians(70)) | (psi <= math.radians(-150)))
    torsion_valid = row["torsion_mask"].numpy().astype(bool).all(-1) & mask
    counts = np.asarray([
        np.sum(helix & torsion_valid),
        np.sum(strand & torsion_valid),
        np.sum((~helix) & (~strand) & torsion_valid),
    ], dtype=np.float64)
    ss = (counts / max(counts.sum(), 1)).tolist()

    contact_pairs = np.argwhere(valid_pair)
    if len(contact_pairs):
        order = np.argsort(distances[contact_pairs[:, 0], contact_pairs[:, 1]])
        contact_pairs = contact_pairs[order[:8]]
    fingerprint = []
    for i, j in contact_pairs:
        scale = max(len(ca) - 1, 1)
        fingerprint.append([int(round(i / scale * 31)), int(round(j / scale * 31))])
    return {
        "length": len(ca),
        "rg": rg,
        "ss": ss,
        "contact_density": density,
        "degree_mean": degree_mean,
        "span_mean": span_mean,
        "shape": shape,
        "fingerprint": fingerprint,
    }


def feature_vector(feature: dict[str, Any]) -> np.ndarray:
    return np.asarray([
        math.log(max(feature["length"], 1)),
        math.log(max(feature["rg"], 1e-4)),
        *feature["ss"],
        feature["contact_density"],
        math.log1p(feature["degree_mean"]),
        feature["span_mean"],
        *feature["shape"],
    ], dtype=np.float64)


def exact_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        abs(left["length"] - right["length"]) / max(left["length"], right["length"]) <= 0.35
        and abs(left["rg"] - right["rg"]) / max(left["rg"], right["rg"], 1e-6) <= 0.45
        and np.abs(np.asarray(left["ss"]) - np.asarray(right["ss"])).sum() <= 0.80
        and abs(left["contact_density"] - right["contact_density"]) <= 0.15
        and np.abs(np.asarray(left["shape"]) - np.asarray(right["shape"])).sum() <= 0.70
    )


def build_negative_index(
    rows: list[dict[str, Any]],
    features: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]] | None = None,
    candidate_features: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], float]:
    candidate_rows = candidate_rows or rows
    candidate_features = candidate_features or features
    matrix = np.stack([feature_vector(value) for value in candidate_features])
    mean = matrix.mean(0, keepdims=True)
    scale = matrix.std(0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    normalized = (matrix - mean) / scale
    target_normalized = (np.stack([feature_vector(value) for value in features]) - mean) / scale
    index = {}
    exact_targets = 0
    for target in range(len(rows)):
        distance = np.sum((normalized - target_normalized[target]) ** 2, axis=1)
        order = [int(i) for i in np.argsort(distance) if candidate_rows[int(i)]["id"] != rows[target]["id"]]
        exact = [i for i in order if exact_match(features[target], candidate_features[i])]
        chosen = (exact + [i for i in order if i not in set(exact)])[:31]
        if len(exact) >= 31:
            exact_targets += 1
        index[rows[target]["id"]] = {
            "negative_ids": [candidate_rows[i]["id"] for i in chosen],
            "exact_negative_count": min(len(exact), 31),
            "nearest_feature_distance": [float(distance[i]) for i in chosen],
        }
    return index, exact_targets / max(len(rows), 1)


def valid_pairs(row: dict[str, Any], minimum: int, maximum: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ca = row["target_coords"].float().numpy()[:, 1]
    mask = row["residue_mask"].numpy().astype(bool)
    sep = np.abs(np.arange(len(ca))[:, None] - np.arange(len(ca))[None, :])
    allowed = mask[:, None] & mask[None, :] & (sep >= minimum)
    if maximum is not None:
        allowed &= sep <= maximum
    i, j = np.where(np.triu(allowed, 1))
    distance = np.linalg.norm(ca[i] - ca[j], axis=1)
    return i, j, distance


def pick_contact(row: dict[str, Any], minimum: int, maximum: int | None, rng: random.Random, desired: bool) -> tuple[int, int, float, bool]:
    left, right, distance = valid_pairs(row, minimum, maximum)
    target = np.flatnonzero(distance < 8.0 if desired else distance >= 12.0)
    if not len(target):
        target = np.arange(len(distance))
    selected = int(rng.choice(target.tolist()))
    value = float(distance[selected])
    return int(left[selected]), int(right[selected]), value, value < 8.0


def pair_text(i: int, j: int) -> str:
    return f"({i + 1},{j + 1})"


def relation_records(
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    feature_by_id: dict[str, dict[str, Any]],
    seed: int,
    balance_index: int | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(stable_seed(row["id"], seed, "relations"))
    sequence = row["sequence"]
    ca = row["target_coords"].float().numpy()[:, 1]
    mask = row["residue_mask"].numpy().astype(bool)
    valid = np.flatnonzero(mask)
    records = []

    def add(operator: str, arguments: list[int], question: str, answer: str, evidence: dict[str, Any], labels: list[str]) -> None:
        records.append({
            "operator": operator,
            "arguments": arguments,
            "question": question,
            "canonical_answer": answer,
            "candidate_labels": labels,
            "evidence": evidence,
        })

    def target_class(slot: int, classes: int) -> int:
        # Use an independently salted operator hash rather than alternating row parity.
        # Alternation balances every marginal perfectly but creates deterministic
        # correlations between labels inside one multi-question assistant turn;
        # under teacher forcing, later labels could then be predicted from an
        # earlier gold label without reading the protein sequence.
        return stable_seed(row["id"], seed, "balanced-target", slot) % classes

    contact_specs = [(6, 11, True, "CONTACT_SHORT"), (12, 23, False, "CONTACT_MEDIUM"), (24, None, True, "CONTACT_LONG")]
    for slot, (minimum, maximum, old_desired, name) in enumerate(contact_specs):
        desired = target_class(slot, 2) == 0 if balance_index is not None else old_desired
        i, j, distance, answer = pick_contact(row, minimum, maximum, rng, desired)
        add(name, [i, j], f"Are residues {i + 1} and {j + 1} within 8 angstroms? A=Yes; B=No.", "A" if answer else "B", {"distance": distance, "threshold": 8.0}, ["A", "B"])

    for slot, (minimum, maximum) in enumerate(((6, 23), (24, None))):
        left, right, distance = valid_pairs(row, minimum, maximum)
        order = np.argsort(distance)
        width = max(1, len(order) // 10)
        low = int(rng.choice(order[:width].tolist()))
        high = int(rng.choice(order[-width:].tolist()))
        if balance_index is not None and target_class(3 + slot, 2) == 1:
            first, second = high, low
        else:
            first, second = low, high
        a = (int(left[first]), int(right[first]), float(distance[first]))
        b = (int(left[second]), int(right[second]), float(distance[second]))
        add(f"DISTANCE_ORDER_{slot + 1}", [a[0], a[1], b[0], b[1]], f"Which pair is closer in the target fold? A={pair_text(a[0], a[1])}; B={pair_text(b[0], b[1])}.", "A" if a[2] < b[2] else "B", {"distance_a": a[2], "distance_b": b[2]}, ["A", "B"])

    starts = [int(i) for i in valid if i + 3 < len(ca) and mask[i : i + 4].all()]
    desired_orientation = ["A", "B", "C"][target_class(5, 3)] if balance_index is not None else None
    start_array = np.asarray(starts, dtype=np.int64)
    segment_vectors = ca[start_array + 3] - ca[start_array]
    segment_vectors /= np.maximum(np.linalg.norm(segment_vectors, axis=1, keepdims=True), 1e-8)
    cosine_matrix = segment_vectors @ segment_vectors.T
    separated = np.abs(start_array[:, None] - start_array[None, :]) >= 8
    if desired_orientation == "A":
        matching = separated & (cosine_matrix >= 0.5)
    elif desired_orientation == "B":
        matching = separated & (cosine_matrix <= -0.5)
    elif desired_orientation == "C":
        matching = separated & (cosine_matrix > -0.5) & (cosine_matrix < 0.5)
    else:
        matching = np.zeros_like(separated)
    available_orientation = np.argwhere(matching)
    if len(available_orientation):
        selected_pair = available_orientation[rng.randrange(len(available_orientation))]
        first_index, second_index = int(selected_pair[0]), int(selected_pair[1])
        first, second = int(start_array[first_index]), int(start_array[second_index])
        cosine = float(cosine_matrix[first_index, second_index])
    else:
        first = rng.choice(starts)
        distant = [i for i in starts if abs(i - first) >= 8] or starts
        second = rng.choice(distant)
        va = ca[first + 3] - ca[first]
        vb = ca[second + 3] - ca[second]
        cosine = float(np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-8))
    orientation = "A" if cosine >= 0.5 else "B" if cosine <= -0.5 else "C"
    add("SEGMENT_ORIENTATION", [first, first + 3, second, second + 3], f"How are segments {first + 1}-{first + 4} and {second + 1}-{second + 4} oriented? A=parallel; B=antiparallel; C=orthogonal-like.", orientation, {"cosine": cosine}, ["A", "B", "C"])

    center = ca[valid].mean(0)
    radii = np.linalg.norm(ca[valid] - center, axis=1)
    near = int(np.argmin(radii)); far = int(np.argmax(radii))
    if rng.random() < 0.5:
        a, b = int(valid[near]), int(valid[far])
    else:
        a, b = int(valid[far]), int(valid[near])
    da, db = float(np.linalg.norm(ca[a] - center)), float(np.linalg.norm(ca[b] - center))
    add("NEARER_CENTER", [a, b], f"Which residue is nearer the fold center? A=residue {a + 1}; B=residue {b + 1}.", "A" if da < db else "B", {"radius_a": da, "radius_b": db}, ["A", "B"])

    frame_candidates = [int(i) for i in valid if np.linalg.norm(ca[i] - row["target_coords"][i, 2].float().numpy()) > 1e-4]
    desired_projection = target_class(6, 2) == 0 if balance_index is not None else None
    frame_array = np.asarray(frame_candidates, dtype=np.int64)
    valid_array = np.asarray(valid, dtype=np.int64)
    axes = row["target_coords"][frame_array, 2].float().numpy() - ca[frame_array]
    axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-8)
    offsets = ca[valid_array][None, :, :] - ca[frame_array][:, None, :]
    projections = np.einsum("fvc,fc->fv", offsets, axes)
    frame_mask = np.abs(frame_array[:, None] - valid_array[None, :]) >= 6
    if desired_projection is not None:
        frame_mask &= (projections >= 0) == desired_projection
    frame_options = np.argwhere(frame_mask)
    if not len(frame_options):
        frame_options = np.argwhere(np.abs(frame_array[:, None] - valid_array[None, :]) >= 6)
    selected_frame = frame_options[rng.randrange(len(frame_options))]
    frame_position, valid_position = int(selected_frame[0]), int(selected_frame[1])
    i, j = int(frame_array[frame_position]), int(valid_array[valid_position])
    projection = float(projections[frame_position, valid_position])
    add("LOCAL_FRAME_DIRECTION", [i, j], f"In residue {i + 1}'s CA-to-C local axis, is residue {j + 1} on the positive side? A=positive; B=negative.", "A" if projection >= 0 else "B", {"signed_projection": projection}, ["A", "B"])

    chirality_starts = [int(i) for i in valid if i + 3 < len(ca) and mask[i : i + 4].all()]
    desired_chirality = target_class(7, 2) == 0 if balance_index is not None else None
    chirality_options = []
    for start in chirality_starts:
        value = float(np.dot(np.cross(ca[start + 1] - ca[start], ca[start + 2] - ca[start]), ca[start + 3] - ca[start]))
        if desired_chirality is None or (value >= 0) == desired_chirality:
            chirality_options.append((start, value))
    if not chirality_options:
        chirality_options = [(start, float(np.dot(np.cross(ca[start + 1] - ca[start], ca[start + 2] - ca[start]), ca[start + 3] - ca[start]))) for start in chirality_starts]
    i, volume = rng.choice(chirality_options)
    add("CA_CHIRALITY", [i, i + 1, i + 2, i + 3], f"What is the signed CA chirality of residues {i + 1}-{i + 4}? A=positive; B=negative.", "A" if volume >= 0 else "B", {"signed_volume": volume}, ["A", "B"])

    retrieval = [row] + candidates[:31]
    rng.shuffle(retrieval)
    lines = []
    target_label = None
    for label, donor in zip(RETRIEVAL_LABELS, retrieval):
        feature = feature_by_id[donor["id"]]
        contacts = ",".join(f"{a}-{b}" for a, b in feature["fingerprint"])
        lines.append(f"{label}: n={feature['length']} rg={feature['rg']:.1f} ss={max(range(3), key=lambda k: feature['ss'][k])} contacts={contacts}")
        if donor["id"] == row["id"]:
            target_label = label
    add("RETRIEVAL_32", [], "Which candidate contact fingerprint matches the sequence?\n" + "\n".join(lines), str(target_label), {"candidate_ids": [value["id"] for value in retrieval], "target_id": row["id"]}, RETRIEVAL_LABELS)

    desired_multi = ["A", "B", "C"][target_class(8, 3)] if balance_index is not None else None
    multi_options: dict[str, list[tuple[int, int, int, float, float]]] = {"A": [], "B": [], "C": []}
    for anchor_candidate in valid:
        anchor_candidate = int(anchor_candidate)
        others = [int(x) for x in valid if abs(int(x) - anchor_candidate) >= 12]
        if len(others) < 2:
            continue
        near = [x for x in others if float(np.linalg.norm(ca[anchor_candidate] - ca[x])) < 8.0]
        far = [x for x in others if float(np.linalg.norm(ca[anchor_candidate] - ca[x])) >= 8.0]
        if near and far:
            near_index, far_index = rng.choice(near), rng.choice(far)
            near_distance = float(np.linalg.norm(ca[anchor_candidate] - ca[near_index]))
            far_distance = float(np.linalg.norm(ca[anchor_candidate] - ca[far_index]))
            multi_options["A"].append((anchor_candidate, near_index, far_index, near_distance, far_distance))
            multi_options["B"].append((anchor_candidate, far_index, near_index, far_distance, near_distance))
        if len(far) >= 2:
            first_far, second_far = rng.sample(far, 2)
            multi_options["C"].append((anchor_candidate, first_far, second_far, float(np.linalg.norm(ca[anchor_candidate] - ca[first_far])), float(np.linalg.norm(ca[anchor_candidate] - ca[second_far]))))
    selected_multi = multi_options.get(desired_multi, []) if desired_multi else []
    if selected_multi:
        anchor, a, b, da, db = rng.choice(selected_multi)
    else:
        anchor = rng.choice(valid.tolist())
        others = [int(x) for x in valid if abs(int(x) - anchor) >= 12]
        a, b = rng.sample(others, 2)
        da, db = float(np.linalg.norm(ca[anchor] - ca[a])), float(np.linalg.norm(ca[anchor] - ca[b]))
    answer = "A" if da < db and da < 8 else "B" if db < da and db < 8 else "C"
    add("MULTI_CONSTRAINT", [anchor, a, b], f"Which option is both closer to residue {anchor + 1} and within 8 angstroms? A=residue {a + 1}; B=residue {b + 1}; C=neither.", answer, {"distance_a": da, "distance_b": db}, ["A", "B", "C"])

    desired_any = target_class(9, 2) == 0 if balance_index is not None else bool(stable_seed(row["id"], "last") % 2)
    i, j, distance, answer = pick_contact(row, 6, None, rng, desired_any)
    add("CONTACT_ANY", [i, j], f"Are residues {i + 1} and {j + 1} within 8 angstroms? A=Yes; B=No.", "A" if answer else "B", {"distance": distance, "threshold": 8.0}, ["A", "B"])
    return records


def ordinary_records(sequence: str, count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    records = []
    for index in range(count):
        if index % 3 == 0:
            i, j = rng.sample(range(len(sequence)), 2)
            answer = "A" if sequence[i] <= sequence[j] else "B"
            question = f"Which residue letter is alphabetically earlier? A=position {i + 1} ({sequence[i]}); B=position {j + 1} ({sequence[j]})."
        elif index % 3 == 1:
            aa = rng.choice(sorted(set(sequence)))
            answer = "A" if sequence.count(aa) % 2 == 0 else "B"
            question = f"Does amino acid {aa} occur an even number of times? A=Yes; B=No."
        else:
            i = rng.randrange(len(sequence))
            answer = "A" if sequence[i] in "AILMFWVY" else "B"
            question = f"Is the residue at position {i + 1} in the set AILMFWVY? A=Yes; B=No."
        records.append({"question": question, "canonical_answer": answer})
    return records


def recompute_spatial_answer(row: dict[str, Any], record: dict[str, Any]) -> str:
    """Independently recompute the canonical label from saved arguments."""
    ca = row["target_coords"].float().numpy()[:, 1]
    operator = record["operator"]
    args = record["arguments"]
    if operator.startswith("CONTACT_"):
        return "A" if float(np.linalg.norm(ca[args[0]] - ca[args[1]])) < 8.0 else "B"
    if operator.startswith("DISTANCE_ORDER_"):
        first = float(np.linalg.norm(ca[args[0]] - ca[args[1]]))
        second = float(np.linalg.norm(ca[args[2]] - ca[args[3]]))
        return "A" if first < second else "B"
    if operator == "SEGMENT_ORIENTATION":
        first = ca[args[1]] - ca[args[0]]
        second = ca[args[3]] - ca[args[2]]
        cosine = float(np.dot(first, second) / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-8))
        return "A" if cosine >= 0.5 else "B" if cosine <= -0.5 else "C"
    if operator == "NEARER_CENTER":
        mask = row["residue_mask"].numpy().astype(bool)
        center = ca[mask].mean(0)
        first = float(np.linalg.norm(ca[args[0]] - center))
        second = float(np.linalg.norm(ca[args[1]] - center))
        return "A" if first < second else "B"
    if operator == "LOCAL_FRAME_DIRECTION":
        axis = row["target_coords"][args[0], 2].float().numpy() - ca[args[0]]
        projection = float(np.dot(ca[args[1]] - ca[args[0]], axis))
        return "A" if projection >= 0 else "B"
    if operator == "CA_CHIRALITY":
        i, j, k, ell = args
        volume = float(np.dot(np.cross(ca[j] - ca[i], ca[k] - ca[i]), ca[ell] - ca[i]))
        return "A" if volume >= 0 else "B"
    if operator == "RETRIEVAL_32":
        position = record["evidence"]["candidate_ids"].index(row["id"])
        return RETRIEVAL_LABELS[position]
    if operator == "MULTI_CONSTRAINT":
        anchor, first_index, second_index = args
        first = float(np.linalg.norm(ca[anchor] - ca[first_index]))
        second = float(np.linalg.norm(ca[anchor] - ca[second_index]))
        return "A" if first < second and first < 8 else "B" if second < first and second < 8 else "C"
    raise ValueError(f"Unsupported operator {operator}")


def render_training(tokenizer: Any, sequence: str, questions: list[str], answers: list[str], system: str) -> tuple[torch.Tensor, torch.Tensor, int]:
    user = f"Protein sequence:\n{sequence}\n\n" + "\n".join(f"Q{index + 1:02d}. {question}" for index, question in enumerate(questions))
    prefix = tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # The first answer is emitted directly after the assistant prefix so the
    # training context matches the one-question evaluation protocol.  Every
    # option label, including all 32 retrieval labels, is a verified one-token
    # target; BL1/BL2/BL3 therefore have identical assistant-token budgets.
    answer_text = " ".join(answers)
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    input_ids = torch.tensor(prefix_ids + answer_ids, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[len(prefix_ids) :] = torch.tensor(answer_ids, dtype=torch.long)
    return input_ids, labels, len(answer_ids)


def eval_prompt(tokenizer: Any, sequence: str, question: str) -> list[int]:
    rendered = tokenizer.apply_chat_template([
        {"role": "system", "content": "Answer the protein question. Return only the option label."},
        {"role": "user", "content": f"Protein sequence:\n{sequence}\n\n{question}"},
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def kmer_set(sequence: str, k: int = 5) -> set[str]:
    return {sequence[i : i + k] for i in range(max(len(sequence) - k + 1, 0))}


def sequence_overlap_audit(science_rows: list[dict[str, Any]], foldbench_rows: list[dict[str, Any]]) -> dict[str, Any]:
    fold_sequences = {row["sequence"] for row in foldbench_rows}
    exact = [row["id"] for row in science_rows if row["sequence"] in fold_sequences]
    fold_kmers = [(row["id"], kmer_set(row["sequence"])) for row in foldbench_rows]
    maxima = []
    for row in science_rows:
        source = kmer_set(row["sequence"])
        best = 0.0
        best_id = None
        for target_id, target in fold_kmers:
            score = len(source & target) / max(len(source | target), 1)
            if score > best:
                best, best_id = score, target_id
        maxima.append((best, row["id"], best_id))
    maxima.sort(reverse=True)
    return {"method": "exact sequence plus exhaustive 5-mer Jaccard", "exact_sequence_overlaps": exact, "maximum_5mer_jaccard": maxima[0][0] if maxima else 0.0, "top_20": [{"score": value, "science_id": source, "foldbench_id": target} for value, source, target in maxima[:20]]}


def main() -> None:
    args = parse_args()
    generator_version = BALANCED_GENERATOR_VERSION if args.balanced_labels else GENERATOR_VERSION
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_cache.parent.mkdir(parents=True, exist_ok=True)
    cache = torch.load(args.geometry_cache, map_location="cpu", weights_only=False)
    foldbench = torch.load(args.foldbench_cache, map_location="cpu", weights_only=False)
    source_splits = {
        source: output
        for source, output in SPLIT_NAMES.items()
        if source in cache["splits"]
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    label_tokens = {label: tokenizer(label, add_special_tokens=False)["input_ids"] for label in RETRIEVAL_LABELS}
    if any(len(value) != 1 for value in label_tokens.values()):
        raise RuntimeError(f"Retrieval labels must be single tokens: {label_tokens}")

    feature_by_id = {}
    row_by_id = {}
    balance_index_by_id = {}
    negative_payload = {"generator_version": generator_version, "splits": {}}
    augmented = {"format_version": 1, "stats": {}, "splits": {}}
    all_records = {}
    qc = {"strict_schema": {"checked": 0, "passed": 0}, "factual_recomputation": {"checked": 0, "passed": 0}, "split_cluster_overlap": {}, "negative_coverage": {}}

    features_by_source_split = {}
    for source_split, output_split in source_splits.items():
        rows = cache["splits"][source_split]
        features = [structure_features(row) for row in rows]
        features_by_source_split[source_split] = features
        for row_index, (row, feature) in enumerate(zip(rows, features)):
            feature_by_id[row["id"]] = feature
            row_by_id[row["id"]] = row
            balance_index_by_id[row["id"]] = row_index

    train_rows = cache["splits"]["train"]
    train_features = features_by_source_split["train"]
    for source_split, output_split in source_splits.items():
        rows = cache["splits"][source_split]
        features = features_by_source_split[source_split]
        if source_split == "train":
            negative_index, coverage = build_negative_index(rows, features)
            pool_name = "train"
        else:
            negative_index, coverage = build_negative_index(
                rows,
                features,
                candidate_rows=train_rows,
                candidate_features=train_features,
            )
            pool_name = "train_only"
        negative_payload["splits"][output_split] = negative_index
        qc["negative_coverage"][output_split] = {"targets": len(rows), "targets_with_31_exact": sum(value["exact_negative_count"] == 31 for value in negative_index.values()), "coverage": coverage, "candidate_count": 31, "candidate_pool": pool_name}

    split_ids = {
        name: {row["source_id"] for row in cache["splits"][source]}
        for source, name in source_splits.items()
    }
    for left in split_ids:
        for right in split_ids:
            if left < right:
                qc["split_cluster_overlap"][f"{left}__{right}"] = sorted(split_ids[left] & split_ids[right])

    for source_split, output_split in source_splits.items():
        rows = cache["splits"][source_split]
        records = []
        augmented_rows = []
        for row in rows:
            negative_ids = negative_payload["splits"][output_split][row["id"]]["negative_ids"]
            candidates = [row_by_id[value] for value in negative_ids]
            relations = relation_records(
                row,
                candidates,
                feature_by_id,
                args.seed,
                balance_index_by_id[row["id"]] if args.balanced_labels else None,
            )
            ordinary = ordinary_records(row["sequence"], len(relations), stable_seed(row["id"], "ordinary", args.seed))
            donor_id = negative_ids[0]
            # Every shuffle donor comes from the train pool.  For dev/test this
            # deliberately avoids using another held-out structure as a label
            # source or distractor.
            donor_negative_ids = negative_payload["splits"]["train"][donor_id]["negative_ids"]
            donor_relations = relation_records(
                row_by_id[donor_id],
                [row_by_id[value] for value in donor_negative_ids],
                feature_by_id,
                args.seed,
                balance_index_by_id[donor_id] if args.balanced_labels else None,
            )
            real_answers = [value["canonical_answer"] for value in relations]
            shuffled_answers = [value["canonical_answer"] for value in donor_relations]
            ordinary_answers = [value["canonical_answer"] for value in ordinary]
            training_order = list(range(len(relations)))
            random.Random(stable_seed(row["id"], "training-order", args.seed)).shuffle(training_order)
            real_pack = render_training(tokenizer, row["sequence"], [relations[index]["question"] for index in training_order], [real_answers[index] for index in training_order], "Answer all protein spatial-relation questions. Use only the requested option labels.")
            shuffled_pack = render_training(tokenizer, row["sequence"], [relations[index]["question"] for index in training_order], [shuffled_answers[index] for index in training_order], "Answer all protein spatial-relation questions. Use only the requested option labels.")
            ordinary_pack = render_training(tokenizer, row["sequence"], [ordinary[index]["question"] for index in training_order], [ordinary_answers[index] for index in training_order], "Answer all protein sequence questions. Use only the requested option labels.")
            assistant_counts = [real_pack[2], shuffled_pack[2], ordinary_pack[2]]
            if len(set(assistant_counts)) != 1:
                raise RuntimeError(f"Assistant token mismatch for {row['id']}: {assistant_counts}")
            copied = dict(row)
            copied["bridge"] = {
                "real": {"input_ids": real_pack[0], "labels": real_pack[1]},
                "shuffled": {"input_ids": shuffled_pack[0], "labels": shuffled_pack[1]},
                "ordinary": {"input_ids": ordinary_pack[0], "labels": ordinary_pack[1]},
                "assistant_tokens": assistant_counts[0],
                "donor_id": donor_id,
                "training_order": training_order,
            }
            augmented_rows.append(copied)
            for index, relation in enumerate(relations):
                record = {
                    "sample_id": f"{row['source_id']}::{relation['operator']}::{index:02d}",
                    "protein_id": row["source_id"],
                    "cache_id": row["id"],
                    "split": output_split,
                    "sequence": row["sequence"],
                    "operator": relation["operator"],
                    "arguments": relation["arguments"],
                    "question": relation["question"],
                    "canonical_answer": relation["canonical_answer"],
                    "shuffled_answer": shuffled_answers[index],
                    "ordinary_question": ordinary[index]["question"],
                    "ordinary_answer": ordinary_answers[index],
                    "candidate_labels": relation["candidate_labels"],
                    "evidence": {**relation["evidence"], "source_structure_id": row["source_id"]},
                    "negative_pool_id": f"{output_split}::{row['id']}",
                    "matched_shuffle_donor_id": donor_id,
                    "matched_shuffle_donor_sequence": row_by_id[donor_id]["sequence"],
                    "generator_version": generator_version,
                    "eval_prompt_ids": eval_prompt(tokenizer, row["sequence"], relation["question"]),
                }
                required = ("sample_id", "protein_id", "split", "sequence", "operator", "question", "canonical_answer", "evidence", "negative_pool_id", "generator_version")
                qc["strict_schema"]["checked"] += 1
                if all(key in record and record[key] not in (None, "") for key in required) and record["canonical_answer"] in record["candidate_labels"]:
                    qc["strict_schema"]["passed"] += 1
                qc["factual_recomputation"]["checked"] += 1
                factual = record["canonical_answer"] == recompute_spatial_answer(row, record)
                if factual:
                    qc["factual_recomputation"]["passed"] += 1
                records.append(record)
        all_records[output_split] = records
        augmented["splits"][source_split] = augmented_rows

    augmented["stats"] = {
        "generator_version": generator_version,
        "geometry_cache": str(args.geometry_cache),
        "seed": args.seed,
        "records": {split: len(records) for split, records in all_records.items()},
        "relation_inputs": {source: len(rows) for source, rows in augmented["splits"].items()},
    }
    for split, records in all_records.items():
        path = args.output_dir / f"{split}.jsonl"
        with path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    json_dump(args.output_dir / "hard_negative_index.json", negative_payload)
    qc["strict_schema"]["rate"] = qc["strict_schema"]["passed"] / max(qc["strict_schema"]["checked"], 1)
    qc["factual_recomputation"]["rate"] = qc["factual_recomputation"]["passed"] / max(qc["factual_recomputation"]["checked"], 1)
    qc["label_balance"] = {}
    for split, records in all_records.items():
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for record in records:
            counts[record["operator"]][record["canonical_answer"]] += 1
        qc["label_balance"][split] = {
            operator: {
                "counts": dict(counter),
                "majority_rate": max(counter.values()) / sum(counter.values()),
            }
            for operator, counter in sorted(counts.items())
        }
    science_rows = [row for split in cache["splits"].values() for row in split]
    qc["foldbench_sequence_overlap"] = sequence_overlap_audit(science_rows, foldbench["splits"]["validation"])
    json_dump(args.output_dir / "recomputation_audit.json", qc)
    torch.save(augmented, args.output_cache)

    split_hashes = {path.name: sha256(path) for path in sorted(args.output_dir.glob("*.jsonl"))}
    split_hashes[args.output_cache.name] = sha256(args.output_cache)
    json_dump(args.output_dir / "split_hashes.json", split_hashes)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        diff = subprocess.check_output(["git", "diff", "--binary"], stderr=subprocess.DEVNULL)
        diff_hash = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        commit, diff_hash = None, None
    manifest = {
        "status": "COMPLETE",
        "generator_version": generator_version,
        "balanced_labels": args.balanced_labels,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "source_cache": str(args.geometry_cache),
        "source_cache_sha256": sha256(args.geometry_cache),
        "foldbench_cache": str(args.foldbench_cache),
        "model": str(args.model),
        "git_commit": commit,
        "dirty_diff_sha256": diff_hash,
        "split_hashes": split_hashes,
        "qc": qc,
        "operator_counts": {split: dict(Counter(record["operator"] for record in records)) for split, records in all_records.items()},
        "label_token_ids": {key: value[0] for key, value in label_tokens.items()},
        "hard_match_definition": {"length_relative": 0.35, "rg_relative": 0.45, "ss_l1": 0.80, "contact_density_absolute": 0.15, "shape_l1": 0.70},
    }
    json_dump(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"status": "COMPLETE", "output": str(args.output_dir), "cache": str(args.output_cache), "qc": qc, "records": augmented["stats"]["records"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
