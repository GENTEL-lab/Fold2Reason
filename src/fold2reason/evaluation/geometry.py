#!/usr/bin/env python3
"""Folding metrics and distributed evaluation for the v2 geometry head."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from fold2reason.data.backbone import to_bb4q10
from fold2reason.models.geometry import (
    GeometryModel,
    add_all_attention_lora,
    load_lora,
    load_text_model,
)
from fold2reason.losses.geometry import geometry_loss
from fold2reason.data.geometry_cache import backbone_torsions


METRIC_NAMES = [
    "loss",
    "tm_score",
    "gdt_ts",
    "gdt_ha",
    "lddt_ca",
    "ca_rmsd",
    "ca_distance_mae",
    "ca_distance_rmse",
    "long_range_ca_distance_mae",
    "contact_precision",
    "contact_recall",
    "contact_f1",
    "contact_top_l_precision",
    "long_contact_precision",
    "long_contact_recall",
    "long_contact_f1",
    "long_contact_top_l5_precision",
    "rg_ratio",
    "clash_per_1000_atoms",
    "clash_residue_fraction",
    "ca_step_valid_fraction",
    "backbone_bond_valid_fraction",
    "torsion_mae_degrees",
]
VIEW_NAMES = ["SEQ", "SEQ_MSA", "SEQ_MSA_TPL"]


def kabsch_align(
    predicted: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_centered = predicted - predicted.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    covariance = predicted_centered.T @ target_centered
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    aligned = predicted_centered @ rotation + target.mean(axis=0)
    distances = np.linalg.norm(aligned - target, axis=-1)
    return aligned, distances


def lddt_ca(predicted: np.ndarray, target: np.ndarray) -> float:
    target_distance = np.linalg.norm(
        target[:, None] - target[None, :], axis=-1
    )
    predicted_distance = np.linalg.norm(
        predicted[:, None] - predicted[None, :], axis=-1
    )
    mask = np.triu(target_distance < 15.0, k=1)
    if not mask.any():
        return 0.0
    error = np.abs(predicted_distance - target_distance)[mask]
    return float(
        np.mean(
            [
                np.mean(error < threshold)
                for threshold in (0.5, 1.0, 2.0, 4.0)
            ]
        )
    )


def distance_map_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    length = len(predicted)
    separation = np.abs(
        np.arange(length)[:, None] - np.arange(length)[None, :]
    )
    upper = np.triu(np.ones((length, length), dtype=bool), k=1)
    predicted_distance = np.linalg.norm(
        predicted[:, None] - predicted[None, :], axis=-1
    )
    target_distance = np.linalg.norm(
        target[:, None] - target[None, :], axis=-1
    )
    error = np.abs(predicted_distance - target_distance)
    pair_error = error[upper]
    long_mask = upper & (separation >= 24)
    long_error = error[long_mask]
    return {
        "ca_distance_mae": float(np.mean(pair_error))
        if pair_error.size
        else 0.0,
        "ca_distance_rmse": float(np.sqrt(np.mean(pair_error**2)))
        if pair_error.size
        else 0.0,
        "long_range_ca_distance_mae": float(np.mean(long_error))
        if long_error.size
        else 0.0,
    }


def contact_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    min_separation: int = 6,
    top_fraction: float = 1.0,
) -> tuple[float, float, float, float]:
    length = len(predicted)
    separation = np.abs(
        np.arange(length)[:, None] - np.arange(length)[None, :]
    )
    upper = np.triu(np.ones((length, length), dtype=bool), k=1)
    valid = upper & (separation >= min_separation)
    predicted_distance = np.linalg.norm(
        predicted[:, None] - predicted[None, :], axis=-1
    )
    target_distance = np.linalg.norm(
        target[:, None] - target[None, :], axis=-1
    )
    predicted_contact = (predicted_distance < 8.0) & valid
    target_contact = (
        target_distance < 8.0
    ) & valid
    predicted_count = int(predicted_contact.sum())
    target_count = int(target_contact.sum())
    overlap = int((predicted_contact & target_contact).sum())
    precision = overlap / predicted_count if predicted_count else 0.0
    recall = overlap / target_count if target_count else 0.0
    if predicted_count + target_count:
        f1 = 2 * overlap / (predicted_count + target_count)
    else:
        f1 = 1.0
    valid_pairs = np.argwhere(valid)
    top_k = min(max(1, int(round(length * top_fraction))), len(valid_pairs))
    if top_k == 0:
        top_precision = 0.0
    else:
        scores = predicted_distance[valid]
        order = np.argpartition(scores, top_k - 1)[:top_k]
        target_flat = target_contact[valid]
        top_precision = float(np.mean(target_flat[order]))
    return precision, recall, f1, top_precision


def clash_metrics(predicted_coords: np.ndarray) -> dict[str, float]:
    atoms = predicted_coords.reshape(-1, 3)
    residue_index = np.repeat(
        np.arange(predicted_coords.shape[0]), predicted_coords.shape[1]
    )
    distance = np.linalg.norm(
        atoms[:, None, :] - atoms[None, :, :], axis=-1
    )
    upper = np.triu(np.ones(distance.shape, dtype=bool), k=1)
    same_or_adjacent_residue = (
        np.abs(residue_index[:, None] - residue_index[None, :]) <= 1
    )
    evaluable = upper & ~same_or_adjacent_residue
    clashes = evaluable & (distance < 2.0)
    clash_count = int(clashes.sum())
    if clash_count:
        clashing_residues = np.unique(
            np.concatenate(
                [
                    residue_index[np.where(clashes)[0]],
                    residue_index[np.where(clashes)[1]],
                ]
            )
        )
        clash_residue_fraction = len(clashing_residues) / len(
            predicted_coords
        )
    else:
        clash_residue_fraction = 0.0
    return {
        "clash_per_1000_atoms": float(
            1000.0 * clash_count / max(len(atoms), 1)
        ),
        "clash_residue_fraction": float(clash_residue_fraction),
    }


def score_coordinates(
    predicted_coords: np.ndarray,
    target_coords: np.ndarray,
    residue_mask: np.ndarray,
) -> dict[str, float]:
    valid = residue_mask.astype(bool)
    predicted_ca = predicted_coords[valid, 1]
    target_ca = target_coords[valid, 1]
    aligned_ca, distances = kabsch_align(predicted_ca, target_ca)
    length = len(distances)
    d0 = max(0.5, 1.24 * np.cbrt(max(length - 15, 1)) - 1.8)
    tm_score = float(
        np.mean(1.0 / (1.0 + (distances / d0) ** 2))
    )
    rmsd = float(np.sqrt(np.mean(distances**2)))
    gdt_ts = float(
        np.mean([np.mean(distances <= t) for t in (1.0, 2.0, 4.0, 8.0)])
    )
    gdt_ha = float(
        np.mean([np.mean(distances <= t) for t in (0.5, 1.0, 2.0, 4.0)])
    )
    precision, recall, f1, top_l_precision = contact_metrics(
        predicted_ca, target_ca, min_separation=6, top_fraction=1.0
    )
    long_precision, long_recall, long_f1, long_top_l5_precision = (
        contact_metrics(
            predicted_ca,
            target_ca,
            min_separation=24,
            top_fraction=0.2,
        )
    )
    distance_metrics = distance_map_metrics(predicted_ca, target_ca)
    clash = clash_metrics(predicted_coords[valid])
    predicted_rg = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (
                        predicted_ca - predicted_ca.mean(axis=0)
                    )
                    ** 2,
                    axis=-1,
                )
            )
        )
    )
    target_rg = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (target_ca - target_ca.mean(axis=0)) ** 2,
                    axis=-1,
                )
            )
        )
    )
    all_predicted_ca = predicted_coords[:, 1]
    steps = np.linalg.norm(np.diff(all_predicted_ca, axis=0), axis=-1)
    predicted_bonds = np.concatenate(
        [
            np.linalg.norm(
                predicted_coords[:, left] - predicted_coords[:, right],
                axis=-1,
            )
            for left, right in ((0, 1), (1, 2), (2, 3))
        ]
        + [
            np.linalg.norm(
                predicted_coords[:-1, 2] - predicted_coords[1:, 0],
                axis=-1,
            )
        ]
    )
    target_bonds = np.concatenate(
        [
            np.linalg.norm(
                target_coords[:, left] - target_coords[:, right],
                axis=-1,
            )
            for left, right in ((0, 1), (1, 2), (2, 3))
        ]
        + [
            np.linalg.norm(
                target_coords[:-1, 2] - target_coords[1:, 0],
                axis=-1,
            )
        ]
    )
    predicted_torsion, torsion_mask = backbone_torsions(
        predicted_coords, residue_mask.astype(bool)
    )
    target_torsion, _ = backbone_torsions(
        target_coords, residue_mask.astype(bool)
    )
    torsion_cosine = np.sum(
        predicted_torsion[torsion_mask] * target_torsion[torsion_mask],
        axis=-1,
    )
    torsion_error_degrees = np.degrees(
        np.arccos(np.clip(torsion_cosine, -1.0, 1.0))
    )
    return {
        "tm_score": tm_score,
        "gdt_ts": gdt_ts,
        "gdt_ha": gdt_ha,
        "lddt_ca": lddt_ca(predicted_ca, target_ca),
        "ca_rmsd": rmsd,
        **distance_metrics,
        "contact_precision": precision,
        "contact_recall": recall,
        "contact_f1": f1,
        "contact_top_l_precision": top_l_precision,
        "long_contact_precision": long_precision,
        "long_contact_recall": long_recall,
        "long_contact_f1": long_f1,
        "long_contact_top_l5_precision": long_top_l5_precision,
        "rg_ratio": predicted_rg / max(target_rg, 1e-8),
        **clash,
        "ca_step_valid_fraction": float(
            np.mean((steps >= 3.3) & (steps <= 4.3))
        ),
        "backbone_bond_valid_fraction": float(
            np.mean(np.abs(predicted_bonds - target_bonds) <= 0.2)
        ),
        "torsion_mae_degrees": float(
            np.mean(torsion_error_degrees)
        ),
    }


@torch.inference_mode()
def predict_row(
    model: torch.nn.Module,
    row: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    input_ids = row["input_ids"].to(
        device=device, dtype=torch.long
    )
    marker_positions = row["marker_positions"].to(
        device=device, dtype=torch.long
    )
    outputs = model(input_ids, marker_positions)
    loss, _ = geometry_loss(outputs, row, device)
    predicted = outputs["coords"].float().cpu().numpy()
    target = row["target_coords"].float().numpy()
    residue_mask = row["residue_mask"].numpy()
    metrics = score_coordinates(predicted, target, residue_mask)
    metrics["loss"] = float(loss)
    return metrics, predicted


@torch.inference_mode()
def evaluate_rows(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    split: str,
    rank: int,
    world_size: int,
    device: torch.device,
    max_examples: int = 0,
) -> dict[str, Any]:
    model.eval()
    selected = rows[:max_examples] if max_examples else rows
    local_rows = selected[rank::world_size]
    overall = torch.zeros(
        len(METRIC_NAMES) + 3,
        dtype=torch.float64,
        device=device,
    )
    by_view = torch.zeros(
        len(VIEW_NAMES),
        len(METRIC_NAMES) + 1,
        dtype=torch.float64,
        device=device,
    )
    started = time.perf_counter()
    for index, row in enumerate(local_rows, 1):
        metrics, _ = predict_row(model, row, device)
        values = torch.tensor(
            [metrics[name] for name in METRIC_NAMES],
            dtype=torch.float64,
            device=device,
        )
        overall[: len(METRIC_NAMES)] += values
        overall[len(METRIC_NAMES)] += 1
        overall[len(METRIC_NAMES) + 1] += len(row["input_ids"])
        overall[len(METRIC_NAMES) + 2] += row["sequence_length"]
        view_index = VIEW_NAMES.index(row["input_view"])
        by_view[view_index, : len(METRIC_NAMES)] += values
        by_view[view_index, len(METRIC_NAMES)] += 1
        if rank == 0 and (
            index == 1
            or index % 10 == 0
            or index == len(local_rows)
        ):
            print(
                f"[geometry-eval:{split}] rank0 "
                f"{index}/{len(local_rows)} id={row['id']}",
                flush=True,
            )
    dist.all_reduce(overall, op=dist.ReduceOp.SUM)
    dist.all_reduce(by_view, op=dist.ReduceOp.SUM)
    elapsed = time.perf_counter() - started
    count = overall[len(METRIC_NAMES)].item()
    result = {
        "split": split,
        "examples": int(count),
        "input_tokens": int(overall[len(METRIC_NAMES) + 1].item()),
        "residues": int(overall[len(METRIC_NAMES) + 2].item()),
        "wall_seconds": elapsed,
        **{
            name: float(overall[index].item() / count)
            for index, name in enumerate(METRIC_NAMES)
        },
        "by_view": {},
    }
    for view_index, view in enumerate(VIEW_NAMES):
        view_count = by_view[view_index, len(METRIC_NAMES)].item()
        if not view_count:
            continue
        result["by_view"][view] = {
            "examples": int(view_count),
            **{
                name: float(
                    by_view[view_index, index].item() / view_count
                )
                for index, name in enumerate(METRIC_NAMES)
            },
        }
    return result


def shuffled_same_length_metrics(
    predictions: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_length: dict[int, list[str]] = defaultdict(list)
    for item in predictions:
        by_length[item["sequence_length"]].append(item["id"])
    tm_matched = []
    tm_shuffled = []
    contact_matched = []
    contact_shuffled = []
    for ids in by_length.values():
        if len(ids) < 2:
            continue
        ordered = sorted(ids)
        for index, item_id in enumerate(ordered):
            other_id = ordered[(index + 1) % len(ordered)]
            item = next(x for x in predictions if x["id"] == item_id)
            row = rows_by_id[item_id]
            other = rows_by_id[other_id]
            predicted = np.asarray(item["predicted_coords"])
            matched = score_coordinates(
                predicted,
                row["target_coords"].numpy(),
                row["residue_mask"].numpy(),
            )
            shuffled = score_coordinates(
                predicted,
                other["target_coords"].numpy(),
                other["residue_mask"].numpy(),
            )
            tm_matched.append(matched["tm_score"])
            tm_shuffled.append(shuffled["tm_score"])
            contact_matched.append(matched["contact_f1"])
            contact_shuffled.append(shuffled["contact_f1"])
    return {
        "examples": len(tm_matched),
        "matched_tm_score": float(np.mean(tm_matched)),
        "shuffled_tm_score": float(np.mean(tm_shuffled)),
        "tm_score_gap": float(
            np.mean(tm_matched) - np.mean(tm_shuffled)
        ),
        "matched_contact_f1": float(np.mean(contact_matched)),
        "shuffled_contact_f1": float(np.mean(contact_shuffled)),
        "contact_f1_gap": float(
            np.mean(contact_matched) - np.mean(contact_shuffled)
        ),
    }


def stored_negative_metrics(
    predictions: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    """Compare each prediction with its matched and cached negative target."""
    matched_tm = []
    negative_tm = []
    matched_contact = []
    negative_contact = []
    for item in predictions:
        row = rows_by_id[item["id"]]
        if field not in row:
            continue
        predicted = np.asarray(item["predicted_coords"])
        mask = row["residue_mask"].numpy()
        matched = score_coordinates(predicted, row["target_coords"].numpy(), mask)
        negative = score_coordinates(predicted, row[field].numpy(), mask)
        matched_tm.append(matched["tm_score"])
        negative_tm.append(negative["tm_score"])
        matched_contact.append(matched["contact_f1"])
        negative_contact.append(negative["contact_f1"])
    if not matched_tm:
        return {"examples": 0}
    return {
        "examples": len(matched_tm),
        "matched_tm_score": float(np.mean(matched_tm)),
        "negative_tm_score": float(np.mean(negative_tm)),
        "tm_score_gap": float(np.mean(matched_tm) - np.mean(negative_tm)),
        "matched_contact_f1": float(np.mean(matched_contact)),
        "negative_contact_f1": float(np.mean(negative_contact)),
        "contact_f1_gap": float(np.mean(matched_contact) - np.mean(negative_contact)),
    }


def summarize_metric_rows(
    metric_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"examples": len(metric_rows)}
    for name in METRIC_NAMES:
        values = np.asarray(
            [item["metrics"][name] for item in metric_rows],
            dtype=np.float64,
        )
        result[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p25": float(np.quantile(values, 0.25)),
            "p75": float(np.quantile(values, 0.75)),
        }
    return result


@torch.inference_mode()
def evaluate_rows_detailed(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    split: str,
    rank: int,
    world_size: int,
    device: torch.device,
    max_examples: int = 0,
    include_predicted_coordinates: bool = False,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]] | None]:
    model.eval()
    selected = rows[:max_examples] if max_examples else rows
    local_predictions = []
    for index, row in enumerate(selected[rank::world_size], 1):
        metrics, predicted = predict_row(model, row, device)
        local_predictions.append(
            {
                "id": row["id"],
                "sequence": row["sequence"],
                "sequence_length": row["sequence_length"],
                "input_view": row["input_view"],
                "length_bin": row["length_bin"],
                "metrics": metrics,
                "predicted_coords": predicted,
            }
        )
        if rank == 0 and (
            index == 1
            or index % 10 == 0
            or index == math.ceil(len(selected) / world_size)
        ):
            print(
                f"[geometry-detail:{split}] rank0 "
                f"{index}/{math.ceil(len(selected) / world_size)}",
                flush=True,
            )
    gathered: list[list[dict[str, Any]] | None] = [
        None for _ in range(world_size)
    ]
    dist.all_gather_object(gathered, local_predictions)
    if rank != 0:
        return None, None
    predictions = sorted(
        [
            item
            for rank_predictions in gathered
            if rank_predictions is not None
            for item in rank_predictions
        ],
        key=lambda item: item["id"],
    )
    row_by_id = {row["id"]: row for row in selected}
    by_view = {
        view: summarize_metric_rows(
            [item for item in predictions if item["input_view"] == view]
        )
        for view in VIEW_NAMES
        if any(item["input_view"] == view for item in predictions)
    }
    length_bins = sorted({item["length_bin"] for item in predictions})
    by_length_bin = {
        length_bin: summarize_metric_rows(
            [
                item
                for item in predictions
                if item["length_bin"] == length_bin
            ]
        )
        for length_bin in length_bins
    }
    shuffle_input = [
        {
            **item,
            "predicted_coords": item["predicted_coords"].tolist(),
        }
        for item in predictions
    ]
    shuffled = shuffled_same_length_metrics(shuffle_input, row_by_id)
    hard_negative = stored_negative_metrics(
        shuffle_input, row_by_id, "hard_negative_coords"
    )
    structured_negative = stored_negative_metrics(
        shuffle_input, row_by_id, "structured_negative_coords"
    )
    finite = [
        bool(np.isfinite(item["predicted_coords"]).all())
        for item in predictions
    ]
    export_success = []
    for item in predictions:
        try:
            to_bb4q10(item["sequence"], item["predicted_coords"])
            export_success.append(True)
        except (TypeError, ValueError, OverflowError):
            export_success.append(False)
    summary = {
        "split": split,
        **summarize_metric_rows(predictions),
        "by_view": by_view,
        "by_length_bin": by_length_bin,
        "same_length_shuffled_control": shuffled,
        "cached_hard_negative_control": hard_negative,
        "structured_negative_control": structured_negative,
        "finite_coordinate_fraction": float(np.mean(finite)),
        "deterministic_export_success_fraction": float(
            np.mean(export_success)
        ),
    }
    serializable = []
    for item in predictions:
        output_item = {
            key: value
            for key, value in item.items()
            if key != "predicted_coords"
        }
        if include_predicted_coordinates:
            output_item["predicted_coords"] = (
                item["predicted_coords"].tolist()
            )
        serializable.append(output_item)
    return summary, serializable


def parse_args() -> argparse.Namespace:
    project = Path.cwd()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3.5-9B"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=project
        / "artifacts/cache"
        / "qwen35_openfold_1k_v2_geometry.pt",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--base-random-heads",
        action="store_true",
        help=(
            "Evaluate the unfine-tuned text tower with deterministic "
            "randomly initialized geometry heads. This is a pipeline "
            "baseline, not a trained folding model."
        ),
    )
    parser.add_argument(
        "--base-head-seed",
        type=int,
        default=20260730,
        help="Seed for --base-random-heads coordinate/distogram heads.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project
        / "outputs"
        / "qwen35_9b_openfold_1k_v2"
        / "geometry_eval.json",
    )
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Report per-example quantiles and exact-length shuffle controls.",
    )
    parser.add_argument(
        "--include-predicted-coordinates",
        action="store_true",
        help=(
            "Include predicted backbone arrays in detailed output for "
            "post-hoc audits such as template-contact copying."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_random_heads and args.checkpoint is not None:
        raise ValueError("--base-random-heads cannot be combined with --checkpoint")
    if not args.base_random_heads and args.checkpoint is None:
        raise ValueError("Either --checkpoint or --base-random-heads is required")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 1:
        raise RuntimeError(f"Expected at least one GPU, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    base, _ = load_text_model(args.model, device)
    if args.base_random_heads:
        torch.manual_seed(args.base_head_seed)
        lora = add_all_attention_lora(
            base,
            rank=16,
            alpha=32,
            dropout=0.05,
            gradient_checkpointing=False,
        )
    else:
        lora = load_lora(
            base, args.checkpoint / "adapter", trainable=False
        )
    coordinate_mode = "free"
    if args.checkpoint is not None and (
        args.checkpoint / "geometry_config.json"
    ).exists():
        config_path = args.checkpoint / "geometry_config.json"
        config = json.loads(config_path.read_text())
        coordinate_mode = config.get("coordinate_mode", "free")
    model = GeometryModel(
        lora,
        hidden_size=int(base.config.hidden_size),
        coordinate_mode=coordinate_mode,
    )
    model.coordinate_head.to(device)
    model.distogram_head.to(device)
    if not args.base_random_heads:
        heads = torch.load(
            args.checkpoint / "geometry_heads.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_head_state_dict(heads)
    model.eval()
    summaries = {}
    per_example = {}
    for split in (
        name
        for name in ("validation", "validation_rare")
        if name in cache["splits"]
    ):
        if args.detailed:
            split_summary, split_rows = evaluate_rows_detailed(
                model,
                cache["splits"][split],
                split,
                rank,
                world_size,
                device,
                max_examples=args.max_examples,
                include_predicted_coordinates=(
                    args.include_predicted_coordinates
                ),
            )
            if rank == 0:
                summaries[split] = split_summary
                per_example[split] = split_rows
        else:
            summaries[split] = evaluate_rows(
                model,
                cache["splits"][split],
                split,
                rank,
                world_size,
                device,
                max_examples=args.max_examples,
            )
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "summaries": summaries,
                    "per_example": per_example if args.detailed else None,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(json.dumps(summaries, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
