#!/usr/bin/env python3
"""Exact 20-draw Pass@k evaluation for General text or FTB-Core."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    LogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)

PROJECT = Path.cwd()

from fold2reason.evaluation.ftb import (  # noqa: E402
    SYSTEM_PROMPT as FTB_SYSTEM,
    load_benchmark,
    render_prompts as render_ftb_prompts,
    score_prediction as score_ftb,
)
from fold2reason.evaluation.general_text import (  # noqa: E402
    SYSTEM as GENERAL_SYSTEM,
    load_all_rows,
    load_base_model,
    score_row as score_general,
)


K_VALUES = (1, 3, 5, 10, 20)
LOGIT_QUANTIZATION_DENOMINATOR = 1024
LOGIT_QUANTIZATION_STEP = 1.0 / LOGIT_QUANTIZATION_DENOMINATOR


class IndependentStreamSampler(LogitsProcessor):
    """Apply the frozen warpers, then sample with one private RNG per row.

    Transformers merges caller-supplied processors before its built-in
    temperature/top-p warpers.  Sampling in a caller processor would therefore
    otherwise use raw logits.  We apply the two registered warpers here and
    force the sampled token; the generate call disables all outer warpers.
    """

    def __init__(
        self,
        seeds: list[int],
        temperature: float,
        top_p: float,
    ):
        self.seeds = seeds
        self.generators: list[torch.Generator] | None = None
        self.temperature = TemperatureLogitsWarper(temperature)
        self.top_p = TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=1)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.generators is None:
            if scores.shape[0] != len(self.seeds):
                raise RuntimeError(
                    f"expanded generation batch={scores.shape[0]} but seeds={len(self.seeds)}"
                )
            self.generators = []
            for seed in self.seeds:
                generator = torch.Generator(device=scores.device)
                generator.manual_seed(seed)
                self.generators.append(generator)
        # A few fused Qwen3.5 kernels can differ by the last floating-point bits
        # across identical model reloads.  Canonicalize logits on-device before
        # warping so private RNG streams remain replayable without a CPU transfer.
        canonical_scores = torch.round(
            scores.float() * LOGIT_QUANTIZATION_DENOMINATOR
        ) / LOGIT_QUANTIZATION_DENOMINATOR
        warped_scores = self.temperature(input_ids, canonical_scores)
        warped_scores = self.top_p(input_ids, warped_scores)
        chosen = []
        probabilities = torch.softmax(warped_scores.float(), dim=-1)
        for row_index, generator in enumerate(self.generators):
            chosen.append(
                torch.multinomial(
                    probabilities[row_index], 1, generator=generator
                )
            )
        token_ids = torch.stack(chosen).view(-1, 1)
        forced = torch.full_like(warped_scores, -float("inf"))
        forced.scatter_(1, token_ids, 0.0)
        return forced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--benchmark", choices=("general", "ftb"), required=True)
    parser.add_argument("--general-benchs-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--draw-batch-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--eval-seed", type=int, default=20260810)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-shard-count", type=int, default=1)
    parser.add_argument("--data-shard-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stream_seed(item_id: str, draw_index: int, eval_seed: int) -> int:
    payload = f"{item_id}\0{draw_index}\0{eval_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.benchmark == "general":
        loader_args = SimpleNamespace(
            general_benchs_root=args.general_benchs_root,
            datasets=[
                "bbh",
                "chembench",
                "chembench4k",
                "graphqa_easy",
                "graphqa_hard",
                "lab_bench",
                "planbench",
                "scibench",
            ],
            max_examples_per_dataset=1_000_000_000,
        )
        rows, manifest = load_all_rows(loader_args)
    else:
        root = args.general_benchs_root / "data/generated/ftb_core/v1"
        rows = load_benchmark(root, examples_per_task=500)
        manifest = {"benchmark": "ftb_core_v1", "examples": len(rows)}
    if args.limit:
        rows = rows[: args.limit]
        manifest["diagnostic_limit"] = args.limit
    global_ids = [row["id"] for row in rows]
    rows = rows[args.data_shard_index :: args.data_shard_count]
    manifest["distributed_data_shard"] = {
        "count": args.data_shard_count,
        "index": args.data_shard_index,
        "global_examples": len(global_ids),
        "shard_examples": len(rows),
        "global_id_sha256": hashlib.sha256("\n".join(global_ids).encode()).hexdigest(),
        "shard_id_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in rows).encode()
        ).hexdigest(),
    }
    return rows, manifest


def render_prompt(tokenizer: Any, benchmark: str, row: dict[str, Any]) -> str:
    if benchmark == "ftb":
        return render_ftb_prompts(tokenizer, [row])[0]
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": GENERAL_SYSTEM},
            {"role": "user", "content": row["prompt"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def score(args: argparse.Namespace, row: dict[str, Any], output: str) -> dict[str, Any]:
    return score_ftb(row, output) if args.benchmark == "ftb" else score_general(row, output)


def canonical_prediction(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return format(value, ".12g")
    normalized = str(value).strip().lower()
    return normalized or None


def pass_at_k(correct: int, draws: int, k: int) -> float:
    if draws - correct < k:
        return 1.0
    return 1.0 - math.comb(draws - correct, k) / math.comb(draws, k)


def item_summary(draws: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(draw["correct"]) for draw in draws)
    predictions = [draw["normalized_answer"] for draw in draws]
    valid = [prediction for prediction in predictions if prediction is not None]
    counts = Counter(valid)
    total_valid = max(len(valid), 1)
    entropy = -sum(
        (count / total_valid) * math.log(count / total_valid)
        for count in counts.values()
    )
    summary: dict[str, Any] = {
        "correct_draws": correct,
        "correct_sample_rate": correct / len(draws),
        "unique_normalized_answers": len(counts),
        "answer_entropy_nats": entropy,
        "invalid_rate": 1.0 - len(valid) / len(draws),
    }
    for k in K_VALUES:
        summary[f"pass_at_{k}"] = pass_at_k(correct, len(draws), k)
        selected = draws[:k]
        vote_counts = Counter(
            draw["normalized_answer"]
            for draw in selected
            if draw["normalized_answer"] is not None
        )
        if vote_counts:
            best_count = max(vote_counts.values())
            voted = sorted(
                value for value, count in vote_counts.items() if count == best_count
            )[0]
            vote_correct = next(
                bool(draw["correct"])
                for draw in selected
                if draw["normalized_answer"] == voted
            )
        else:
            vote_correct = False
        summary[f"majority_vote_at_{k}"] = vote_correct
    summary["coverage_gap"] = summary["pass_at_20"] - summary["pass_at_1"]
    return summary


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    base = load_base_model(args.model, device)
    if args.adapter is None:
        return base
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    return model


@torch.inference_mode()
def evaluate_row(
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer: Any,
    row: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    prompt = render_prompt(tokenizer, args.benchmark, row)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=32768,
        add_special_tokens=False,
    ).to(device)
    prompt_width = encoded.input_ids.shape[1]
    draw_records = []
    for start in range(0, args.draws, args.draw_batch_size):
        indices = list(range(start, min(start + args.draw_batch_size, args.draws)))
        seeds = [stream_seed(row["id"], index, args.eval_seed) for index in indices]
        sampler = IndependentStreamSampler(
            seeds,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            num_return_sequences=len(indices),
            logits_processor=[sampler],
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(
            generated[:, prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for draw_index, seed, output in zip(indices, seeds, decoded):
            scored = score(args, row, output)
            draw_records.append(
                {
                    "draw_index": draw_index,
                    "stream_seed": seed,
                    "raw_output": output,
                    "normalized_answer": canonical_prediction(scored.get("prediction")),
                    "correct": bool(scored["correct"]),
                    "score": float(scored.get("score", bool(scored["correct"]))),
                }
            )
    draw_records.sort(key=lambda draw: draw["draw_index"])
    if args.benchmark == "general":
        choice_count = len(row.get("choices", [])) or None
    else:
        fixed_choices = {
            "proper_rigid_equivalence": 2,
            "sphere_collision": 2,
            "straight_path_clearance": 2,
            "tetrahedral_chirality": 2,
            "weighted_contact_evidence": 2,
            "sparse_constraints_candidate_selection": 4,
            "fragment_interface_assembly": 4,
            "noisy_template_selection": 4,
        }
        choice_count = fixed_choices.get(row["task"])
        if row["task"] == "nearest_neighbor":
            choice_count = int(row.get("difficulty", {}).get("n_points", 0)) or None
    record = {
        "id": row["id"],
        "benchmark": args.benchmark,
        "dataset": row.get("dataset", "ftb_core"),
        "category": row.get("category", row.get("task", "unknown")),
        "family": row.get("family"),
        "choice_count": choice_count,
        "draws": draw_records,
        **item_summary(draw_records),
    }
    return record


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [f"pass_at_{k}" for k in K_VALUES] + [
        f"majority_vote_at_{k}" for k in K_VALUES
    ] + [
        "coverage_gap",
        "correct_sample_rate",
        "unique_normalized_answers",
        "answer_entropy_nats",
        "invalid_rate",
    ]

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "examples": len(group),
            **{
                name: sum(float(row[name]) for row in group) / len(group)
                for name in metric_names
            },
        }
        option_counts = {row["choice_count"] for row in group}
        if len(option_counts) == 1 and None not in option_counts:
            options = int(next(iter(option_counts)))
            payload["random_choice_chance"] = 1 / options
            payload["chance_adjusted_pass_at_k"] = {}
            for k in K_VALUES:
                chance_pass = 1.0 - (1.0 - 1.0 / options) ** k
                observed = payload[f"pass_at_{k}"]
                payload["chance_adjusted_pass_at_k"][str(k)] = (
                    observed - chance_pass
                ) / max(1.0 - chance_pass, 1e-12)
        return payload

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_choice_count: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_dataset[row["dataset"]].append(row)
        by_choice_count[str(row["choice_count"])].append(row)
    return {
        "overall": summarize(records),
        "by_dataset": {
            name: summarize(group) for name, group in sorted(by_dataset.items())
        },
        "by_choice_count": {
            name: summarize(group)
            for name, group in sorted(by_choice_count.items())
        },
    }


def main() -> None:
    args = parse_args()
    if args.draws != 20 or any(k > args.draws for k in K_VALUES):
        raise ValueError("primary contract requires exactly 20 draws")
    if args.max_new_tokens is None:
        args.max_new_tokens = 128 if args.benchmark == "general" else 32
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed_all(args.eval_seed)
    rows, manifest = load_rows(args)
    local_rows = rows[rank::world_size]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = "left"
    model = load_model(args, device)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    output_path = args.output / f"rank-{rank}.jsonl"
    with output_path.open("w") as handle:
        for index, row in enumerate(local_rows, 1):
            record = evaluate_row(args, model, tokenizer, row, device)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            if rank == 0 and (index == 1 or index % 25 == 0 or index == len(local_rows)):
                print(f"[pass@k:{args.benchmark}] rank0 {index}/{len(local_rows)}", flush=True)
    elapsed = torch.tensor(time.perf_counter() - started, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    dist.barrier()
    if rank == 0:
        records = []
        for source in sorted(args.output.glob("rank-*.jsonl")):
            with source.open() as handle:
                records.extend(json.loads(line) for line in handle if line.strip())
        records.sort(key=lambda row: row["id"])
        summary = {
            "status": "COMPLETE",
            "model_name": args.model_name,
            "model": str(args.model),
            "adapter": str(args.adapter) if args.adapter else None,
            "benchmark": args.benchmark,
            "generation_contract": {
                "draws": args.draws,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "eval_seed": args.eval_seed,
                "seed_derivation": "SHA256(item_id, draw_index, eval_seed)",
                "max_new_tokens": args.max_new_tokens,
                "draw_batch_size": args.draw_batch_size,
                "logit_quantization_step": LOGIT_QUANTIZATION_STEP,
                "model_construction_seed": args.eval_seed,
            },
            "manifest": manifest,
            "wall_seconds_max_rank": float(elapsed.cpu()),
            **aggregate(records),
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
