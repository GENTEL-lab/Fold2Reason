#!/usr/bin/env python3
"""Evaluate Phase-2 entity-memory interventions on FoldBench monomers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from fold2reason.evaluation.geometry import METRIC_NAMES, score_coordinates
from fold2reason.models.geometry import load_lora, load_text_model
from fold2reason.models.workspace import WorkspaceGeometryModel
from fold2reason.training.geometry import json_dump, setup_distributed


def parse_args() -> argparse.Namespace:
    project = Path.cwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--frozen-decoder-checkpoint",
        type=Path,
        help=(
            "Node-local path override for checkpoints whose serialized config "
            "contains an equivalent decoder path from another shared-storage mount."
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=project / "artifacts/cache/qwen35_foldbench_monomer_geometry.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-examples", type=int, default=0)
    return parser.parse_args()


@torch.inference_mode()
def entity_memory(
    model: WorkspaceGeometryModel,
    row: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    ids = row["input_ids"].to(device=device, dtype=torch.long)
    positions = row["marker_positions"].to(device=device, dtype=torch.long)
    outputs = model.text_tower()(input_ids=ids.unsqueeze(0), use_cache=False)
    hidden = outputs.last_hidden_state[0, positions]
    return model.workspace(hidden)["entity_memory"]


def resize_entities(value: torch.Tensor, length: int) -> torch.Tensor:
    if len(value) == length:
        return value
    return F.interpolate(
        value.transpose(0, 1).unsqueeze(0),
        size=length,
        mode="linear",
        align_corners=False,
    )[0].transpose(0, 1)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [name for name in METRIC_NAMES if name != "loss"]
    result: dict[str, Any] = {"examples": len(rows)}
    for name in metric_names:
        values = np.asarray([row["metrics"][name] for row in rows])
        result[name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p25": float(np.quantile(values, 0.25)),
            "p75": float(np.quantile(values, 0.75)),
        }
    return result


def paired_gap(
    matched: list[dict[str, Any]],
    control: list[dict[str, Any]],
) -> dict[str, Any]:
    control_by_id = {row["id"]: row for row in control}
    metrics = [
        "tm_score",
        "lddt_ca",
        "gdt_ts",
        "contact_f1",
        "long_contact_f1",
        "ca_distance_mae",
    ]
    return {
        "examples": len(matched),
        **{
            name: float(
                np.mean(
                    [
                        row["metrics"][name]
                        - control_by_id[row["id"]]["metrics"][name]
                        for row in matched
                    ]
                )
            )
            for name in metrics
        },
    }


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size, device = setup_distributed()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    rows = cache["splits"].get("test") or cache["splits"].get("validation")
    if args.max_examples:
        rows = rows[: args.max_examples]
    # Nearest-length cyclic donor is deterministic and never shares a target.
    order = sorted(range(len(rows)), key=lambda index: (rows[index]["sequence_length"], rows[index]["id"]))
    donor_index = {
        index: order[(position + 1) % len(order)]
        for position, index in enumerate(order)
    }

    base, loading = load_text_model(args.model, device)
    lora = load_lora(base, args.checkpoint / "adapter", trainable=False)
    state = torch.load(args.checkpoint / "workspace.pt", map_location="cpu", weights_only=False)
    config = state["config"]
    frozen_decoder_checkpoint = (
        args.frozen_decoder_checkpoint
        if args.frozen_decoder_checkpoint is not None
        else Path(config["frozen_decoder_checkpoint"])
    )
    model = WorkspaceGeometryModel(
        lora,
        frozen_decoder_checkpoint,
        workspace_width=config["workspace_width"],
        memory_tokens=config["memory_tokens"],
        fingerprint_dim=config["fingerprint_dim"],
        max_sparse_pairs=config["max_sparse_pairs"],
        relation_uses_memory=config["relation_uses_memory"],
        retrieval_temperature=config["retrieval_temperature"],
    )
    model.load_workspace_state_dict(state)
    model.coordinate_head.to(device)
    model.distogram_head.to(device)
    model.workspace.to(device)
    model.eval()

    local = {condition: [] for condition in ("matched", "zero", "shuffled")}
    for local_count, index in enumerate(range(rank, len(rows), world_size), 1):
        row = rows[index]
        donor = rows[donor_index[index]]
        matched_entity = entity_memory(model, row, device)
        donor_entity = entity_memory(model, donor, device)
        shuffled_entity = resize_entities(donor_entity, len(matched_entity))
        conditions = {
            "matched": matched_entity,
            "zero": torch.zeros_like(matched_entity),
            "shuffled": shuffled_entity,
        }
        target = row["target_coords"].float().numpy()
        mask = row["residue_mask"].numpy()
        for condition, entities in conditions.items():
            predicted = model.coordinate_head(entities).float().cpu().numpy()
            local[condition].append(
                {
                    "id": row["id"],
                    "sequence_length": row["sequence_length"],
                    "donor_id": donor["id"],
                    "donor_sequence_length": donor["sequence_length"],
                    "metrics": score_coordinates(predicted, target, mask),
                }
            )
        if rank == 0 and (
            local_count == 1 or local_count % 10 == 0
        ):
            print(
                f"[workspace-foldbench] rank0 {local_count}/"
                f"{(len(rows) + world_size - 1) // world_size} {row['id']}",
                flush=True,
            )

    gathered: list[dict[str, list[dict[str, Any]]] | None] = [
        None for _ in range(world_size)
    ]
    dist.all_gather_object(gathered, local)
    if rank == 0:
        combined = {
            condition: sorted(
                [
                    row
                    for rank_rows in gathered
                    if rank_rows is not None
                    for row in rank_rows[condition]
                ],
                key=lambda row: row["id"],
            )
            for condition in local
        }
        summaries = {
            condition: summarize(values)
            for condition, values in combined.items()
        }
        exact_length = float(
            np.mean(
                [
                    row["sequence_length"] == row["donor_sequence_length"]
                    for row in combined["shuffled"]
                ]
            )
        )
        payload = {
            "status": "COMPLETE",
            "label": args.label,
            "model": str(args.model),
            "checkpoint": str(args.checkpoint),
            "frozen_decoder_checkpoint": str(frozen_decoder_checkpoint),
            "cache": str(args.cache),
            "model_loading": loading,
            "examples": len(rows),
            "shuffled_control_definition": (
                "nearest-length other FoldBench protein entity memory; linear residue-axis "
                "resampling when lengths differ"
            ),
            "shuffled_exact_length_fraction": exact_length,
            "summaries": summaries,
            "paired_gaps": {
                "matched_minus_zero": paired_gap(combined["matched"], combined["zero"]),
                "matched_minus_shuffled": paired_gap(
                    combined["matched"], combined["shuffled"]
                ),
            },
            "per_condition": combined,
        }
        json_dump(args.output, payload)
        print(json.dumps({"summaries": summaries, "paired_gaps": payload["paired_gaps"]}, indent=2, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
