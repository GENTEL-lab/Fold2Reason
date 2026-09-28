#!/usr/bin/env python3
"""Evaluate Pure-LoRA adapters on frozen relation prompts without memory."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from fold2reason.models.geometry import load_lora, load_text_model
from fold2reason.training.geometry import json_dump


def parse_args() -> argparse.Namespace:
    project = Path.cwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=project / "artifacts/cache/openfold_phase2_workspace_v0.pt",
    )
    parser.add_argument(
        "--relations",
        type=Path,
        default=project / "data/openfold_spatial_bridge_v2_2_independent/dev.jsonl",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-proteins", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conditions", default="matched,zero,shuffled")
    return parser.parse_args()


def summarize_relation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"examples": 0}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["operator"]].append(row)
    by_protein: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_protein[row["cache_id"]].append(row)
    ordinary = [row for row in rows if row["operator"] != "RETRIEVAL_32"]
    target_labels = defaultdict(int)
    prediction_labels = defaultdict(int)
    for row in rows:
        target_labels[row["target"]] += 1
        prediction_labels[row["prediction"]] += 1
    return {
        "examples": len(rows),
        "accuracy": float(np.mean([row["correct"] for row in rows])),
        "candidate_constrained_accuracy": float(
            np.mean([row["candidate_correct"] for row in rows])
        ),
        "ordinary_11_operator_macro_accuracy": float(
            np.mean(
                [
                    np.mean([row["correct"] for row in values])
                    for operator, values in grouped.items()
                    if operator != "RETRIEVAL_32"
                ]
            )
        ),
        "ordinary_11_examples": len(ordinary),
        "protein_exact_all_12": float(
            np.mean([all(row["correct"] for row in values) for values in by_protein.values()])
        ),
        "protein_examples": len(by_protein),
        "answer_label_calibration": {
            "target_counts": dict(sorted(target_labels.items())),
            "greedy_prediction_counts": dict(sorted(prediction_labels.items())),
        },
        "mean_target_margin": float(np.mean([row["target_margin"] for row in rows])),
        "by_operator": {
            operator: {
                "examples": len(values),
                "accuracy": float(np.mean([row["correct"] for row in values])),
                "mean_target_margin": float(
                    np.mean([row["target_margin"] for row in values])
                ),
            }
            for operator, values in sorted(grouped.items())
        },
    }


@torch.inference_mode()
def score_relation_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    label_token: dict[str, int],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    output_rows = []
    records = sorted(records, key=lambda value: len(value["eval_prompt_ids"]))
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        lengths = [len(record["eval_prompt_ids"]) for record in batch]
        maximum = max(lengths)
        ids = torch.full(
            (len(batch), maximum),
            tokenizer.pad_token_id or tokenizer.eos_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(ids)
        for index, record in enumerate(batch):
            values = torch.tensor(
                record["eval_prompt_ids"], dtype=torch.long, device=device
            )
            ids[index, : len(values)] = values
            attention_mask[index, : len(values)] = 1
        output = model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits.float()
        last_indices = torch.tensor(lengths, device=device) - 1
        for index, record in enumerate(batch):
            next_token_logits = logits[index, last_indices[index]]
            labels = record["candidate_labels"]
            scores = np.asarray(
                [float(next_token_logits[label_token[label]]) for label in labels]
            )
            candidate_prediction_index = int(np.argmax(scores))
            target_index = labels.index(record["canonical_answer"])
            greedy_token_id = int(torch.argmax(next_token_logits).item())
            target_token_id = label_token[record["canonical_answer"]]
            candidate_by_token = {label_token[label]: label for label in labels}
            greedy_prediction = candidate_by_token.get(
                greedy_token_id,
                tokenizer.decode([greedy_token_id], skip_special_tokens=False),
            )
            other = np.delete(scores, target_index)
            output_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "cache_id": record["cache_id"],
                    "operator": record["operator"],
                    "target": record["canonical_answer"],
                    "prediction": greedy_prediction,
                    "prediction_token_id": greedy_token_id,
                    "candidate_prediction": labels[candidate_prediction_index],
                    "correct": bool(greedy_token_id == target_token_id),
                    "candidate_correct": bool(candidate_prediction_index == target_index),
                    "target_margin": float(scores[target_index] - np.max(other)),
                    "candidate_scores": {
                        label: float(score) for label, score in zip(labels, scores)
                    },
                }
            )
    return output_rows


def main() -> None:
    args = parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    rows = list(cache["splits"][args.split])
    if args.max_proteins:
        rows = rows[: args.max_proteins]
    selected_ids = {row["id"] for row in rows}
    relation_records = [
        json.loads(line)
        for line in args.relations.open()
        if line.strip()
    ]
    relation_records = [
        record for record in relation_records if record["cache_id"] in selected_ids
    ]
    records_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in relation_records:
        records_by_id[record["cache_id"]].append(record)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    label_token = {}
    for record in relation_records:
        for label in record["candidate_labels"]:
            ids = tokenizer(label, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise RuntimeError(f"Option {label!r} is not one token: {ids}")
            label_token[label] = ids[0]

    device = torch.device("cuda", 0)
    base, loading = load_text_model(args.model, device)
    model = load_lora(base, args.checkpoint / "adapter", trainable=False)
    model.eval()

    adapter_rows = []
    for index, row in enumerate(rows, 1):
        adapter_rows.extend(
            score_relation_batch(
                model,
                tokenizer,
                records_by_id[row["id"]],
                label_token,
                device,
                args.batch_size,
            )
        )
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(f"[pure-lora-relation-eval] {index}/{len(rows)} {row['id']}", flush=True)
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    relation_results = {condition: list(adapter_rows) for condition in conditions}
    relation_summaries = {
        condition: summarize_relation(values)
        for condition, values in relation_results.items()
    }
    matched_accuracy = relation_summaries.get("matched", {}).get("accuracy")
    relation_gaps = {}
    if matched_accuracy is not None:
        for condition in conditions:
            if condition == "matched":
                continue
            value = relation_summaries.get(condition, {}).get("accuracy")
            if value is not None:
                relation_gaps[f"matched_minus_{condition}_accuracy"] = (
                    matched_accuracy - value
                )
    payload = {
        "status": "COMPLETE",
        "label": args.label,
        "model": str(args.model),
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "relations": str(args.relations),
        "split": args.split,
        "protein_examples": len(rows),
        "relation_examples_per_condition": len(adapter_rows),
        "primary_relation_scoring": (
            "full-vocabulary greedy next-token exact match to the one-token canonical label"
        ),
        "supplementary_relation_scoring": (
            "argmax restricted to candidate label tokens"
        ),
        "relation_uses_memory": False,
        "condition_semantics": (
            "Pure-LoRA has no memory input; matched/zero/shuffled are identical "
            "adapter-only scores for compatibility with workspace diagnostics."
        ),
        "model_loading": loading,
        "relation_summaries": relation_summaries,
        "relation_gaps": relation_gaps,
        "per_condition": {"relation": relation_results},
    }
    json_dump(args.output, payload)
    print(
        json.dumps(
            {
                "relation_summaries": relation_summaries,
                "relation_gaps": relation_gaps,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
