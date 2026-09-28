#!/usr/bin/env python3
"""Paired bootstrap and exact McNemar analysis for FTB evaluator shards."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def read_shards(root: Path, mode: str) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted(root.glob(f"{mode}-rank-*.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["id"] in rows:
                    raise ValueError(f"Duplicate id: {row['id']}")
                rows[row["id"]] = row
    return rows


def exact_mcnemar_pvalue(base_only: int, lora_only: int) -> float:
    discordant = base_only + lora_only
    if discordant == 0:
        return 1.0
    tail = min(base_only, lora_only)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(index + 1)
        - math.lgamma(discordant - index + 1)
        - discordant * math.log(2.0)
        for index in range(tail + 1)
    ]
    maximum = max(log_terms)
    lower_tail = math.exp(maximum) * sum(
        math.exp(term - maximum) for term in log_terms
    )
    return min(1.0, 2.0 * lower_tail)


def paired_summary(
    items: list[tuple[bool, bool]],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    pairs = np.asarray(items, dtype=np.float64)
    differences = pairs[:, 1] - pairs[:, 0]
    base_only = int(np.sum((pairs[:, 0] == 1) & (pairs[:, 1] == 0)))
    lora_only = int(np.sum((pairs[:, 0] == 0) & (pairs[:, 1] == 1)))
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    # Bound the temporary resampling-index matrix.  A fixed batch of 250 uses
    # ~139 MB for a 69,528-example full Text8 evaluation and can OOM otherwise
    # lightweight aggregation jobs.
    bootstrap_batch = max(1, min(250, 1_000_000 // len(differences)))
    for start in range(0, bootstrap_samples, bootstrap_batch):
        count = min(bootstrap_batch, bootstrap_samples - start)
        indices = rng.integers(
            0, len(differences), size=(count, len(differences))
        )
        bootstrap[start : start + count] = differences[indices].mean(
            axis=1
        )
    return {
        "examples": len(items),
        "base_accuracy": float(pairs[:, 0].mean()),
        "lora_accuracy": float(pairs[:, 1].mean()),
        "delta_lora_minus_base": float(differences.mean()),
        "paired_bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "both_correct": int(
            np.sum((pairs[:, 0] == 1) & (pairs[:, 1] == 1))
        ),
        "both_wrong": int(
            np.sum((pairs[:, 0] == 0) & (pairs[:, 1] == 0))
        ),
        "base_only_correct": base_only,
        "lora_only_correct": lora_only,
        "mcnemar_exact_two_sided_p": exact_mcnemar_pvalue(
            base_only, lora_only
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = read_shards(args.evaluation_dir, "base")
    lora = read_shards(args.evaluation_dir, "lora")
    if set(base) != set(lora):
        raise ValueError("Base and LoRA ids do not match")
    ids = sorted(base)
    rng = np.random.default_rng(args.seed)
    overall = paired_summary(
        [
            (bool(base[item_id]["correct"]), bool(lora[item_id]["correct"]))
            for item_id in ids
        ],
        rng,
        args.bootstrap_samples,
    )
    grouped: dict[str, dict[str, list[str]]] = {
        "split": defaultdict(list),
        "family": defaultdict(list),
        "task": defaultdict(list),
    }
    for item_id in ids:
        for key in grouped:
            grouped[key][base[item_id][key]].append(item_id)
    payload = {
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "overall": overall,
        "by_split": {},
        "by_family": {},
        "by_task": {},
    }
    for key, groups in grouped.items():
        output_key = f"by_{key}"
        for name, group_ids in sorted(groups.items()):
            payload[output_key][name] = paired_summary(
                [
                    (
                        bool(base[item_id]["correct"]),
                        bool(lora[item_id]["correct"]),
                    )
                    for item_id in group_ids
                ],
                rng,
                args.bootstrap_samples,
            )
    output = args.output or args.evaluation_dir / "paired_stats.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
