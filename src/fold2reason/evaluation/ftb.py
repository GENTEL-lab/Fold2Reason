#!/usr/bin/env python3
"""Stratified base-vs-LoRA evaluation on the local FTB-Core benchmark."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM

from fold2reason.models.backbones import (
    is_ministral3_base_tokenizer,
    is_mllama_base_tokenizer,
    load_internvl35_text_model,
    load_ministral3_text_model,
    load_model_level_tokenizer,
    local_model_type,
    render_ministral3_base_completion,
    render_mllama_base_completion,
)


SYSTEM_PROMPT = (
    "You are a deterministic spatial benchmark answer function. Do not "
    "explain, reason, restate the question, or add prose. Emit exactly the "
    "requested answer and stop."
)


def parse_args() -> argparse.Namespace:
    project = Path.cwd()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3.5-9B"),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=project
        / "outputs"
        / "qwen35_9b_openfold_highconf1k_lora_v1"
        / "final",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path(
            "data/benchmarks/data/generated/ftb_core/v1"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project
        / "outputs"
        / "qwen35_9b_openfold_highconf1k_lora_v1"
        / "ftb_core_eval",
    )
    parser.add_argument("--examples-per-task", type=int, default=50)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Exact ftb-mini-manifest-v1 IDs; overrides --examples-per-task.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--base-shards-from",
        type=Path,
        help=(
            "Reuse base-rank-*.jsonl from an identical prior selection. "
            "IDs are checked before use."
        ),
    )
    parser.add_argument(
        "--lora-shards-from",
        type=Path,
        help=(
            "Reuse lora-rank-*.jsonl from an identical prior selection. "
            "IDs are checked before use."
        ),
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("both", "lora-only"),
        default="both",
        help=(
            "lora-only writes adapter shards without a summary so adapter "
            "inference can run before compatible Base shards arrive."
        ),
    )
    return parser.parse_args()


def load_benchmark(
    root: Path,
    examples_per_task: int,
    selection_manifest: Path | None = None,
) -> list[dict[str, Any]]:
    available = []
    for split in ("test", "test_ood"):
        with gzip.open(root / f"{split}.jsonl.gz", "rt") as handle:
            for line in handle:
                available.append(json.loads(line))
    if selection_manifest is not None:
        manifest = json.loads(selection_manifest.read_text())
        if manifest.get("format") != "ftb-mini-manifest-v1":
            raise ValueError(f"Unsupported selection manifest: {selection_manifest}")
        requested = {
            str(item_id)
            for stratum in manifest["strata"].values()
            for item_id in stratum["ids"]
        }
        if len(requested) != int(manifest["total_examples"]):
            raise ValueError("Selection manifest has duplicate or inconsistent IDs")
        observed_hash = hashlib.sha256("\n".join(sorted(requested)).encode()).hexdigest()
        if observed_hash != manifest["global_id_sha256"]:
            raise ValueError("Selection manifest global ID hash does not match")
        by_id = {str(row["id"]): row for row in available}
        missing = requested - set(by_id)
        if missing:
            raise ValueError(f"Selection manifest has {len(missing)} absent IDs")
        return [by_id[item_id] for item_id in sorted(requested)]

    selected = []
    by_split_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in available:
        by_split_task[(row["split"], row["task"])].append(row)
    for split in ("test", "test_ood"):
        by_task = {
            task: rows
            for (row_split, task), rows in by_split_task.items()
            if row_split == split
        }
        for task in sorted(by_task):
            rows = sorted(by_task[task], key=lambda row: row["id"])
            selected.extend(rows[:examples_per_task])
    return sorted(selected, key=lambda row: row["id"])


def load_model(
    model_path: Path,
    adapter_path: Path,
    device: torch.device,
) -> PeftModel:
    if local_model_type(model_path) == "internvl_chat":
        base = load_internvl35_text_model(model_path, device)
        model = PeftModel.from_pretrained(
            base, adapter_path, is_trainable=False
        )
        model.eval()
        return model
    if local_model_type(model_path) == "mistral3":
        base = load_ministral3_text_model(model_path, device)
        model = PeftModel.from_pretrained(
            base, adapter_path, is_trainable=False
        )
        model.eval()
        return model
    raw_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    text_config = getattr(raw_config, "text_config", None)
    if getattr(text_config, "model_type", None) == "qwen3_5":
        text_config._attn_implementation = "sdpa"
        base = Qwen3_5ForCausalLM.from_pretrained(
            model_path,
            config=text_config,
            key_mapping={r"^model\.language_model\.": "model."},
            dtype=torch.bfloat16,
            device_map={"": device.index},
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    else:
        raw_config._attn_implementation = "sdpa"
        raw_config.use_cache = False
        if text_config is not None:
            text_config._attn_implementation = "sdpa"
            text_config.use_cache = False
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=raw_config,
            dtype=torch.bfloat16,
            device_map={"": device.index},
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    model = PeftModel.from_pretrained(
        base, adapter_path, is_trainable=False
    )
    model.eval()
    return model


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).strip(" `\"'.:;").lower()


def valid_token_id(tokenizer: Any, token: str) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if not isinstance(token_id, int) or token_id < 0:
        return None
    if token_id == getattr(tokenizer, "unk_token_id", None):
        return None
    return token_id


def has_gemma4_turn_tokens(tokenizer: Any) -> bool:
    return (
        valid_token_id(tokenizer, "<|turn>") is not None
        and valid_token_id(tokenizer, "<turn|>") is not None
    )


def generation_eos_token_ids(tokenizer: Any) -> int | list[int]:
    eos_ids = [int(tokenizer.eos_token_id)]
    if getattr(tokenizer, "chat_template", None) and has_gemma4_turn_tokens(tokenizer):
        for token in ("<turn|>", getattr(tokenizer, "str_token", None)):
            if not token:
                continue
            token_id = valid_token_id(tokenizer, str(token))
            if token_id is not None and token_id not in eos_ids:
                eos_ids.append(token_id)
    return eos_ids[0] if len(eos_ids) == 1 else eos_ids


def extract_categorical(text: str, task: str) -> str | None:
    normalized = normalize_text(text)
    allowed = {
        "proper_rigid_equivalence": ["yes", "no"],
        "sphere_collision": ["yes", "no"],
        "straight_path_clearance": ["yes", "no"],
        "tetrahedral_chirality": ["positive", "negative"],
        "weighted_contact_evidence": ["no_contact", "contact"],
        "sparse_constraints_candidate_selection": ["a", "b", "c", "d"],
        "fragment_interface_assembly": ["a", "b", "c", "d"],
        "noisy_template_selection": ["a", "b", "c", "d"],
    }.get(task)
    if allowed is not None:
        canonical = normalized.replace("no contact", "no_contact")
        for label in sorted(allowed, key=len, reverse=True):
            if re.search(
                rf"(?<![a-z0-9_]){re.escape(label)}(?![a-z0-9_])",
                canonical,
            ):
                return label
        return None
    if task == "nearest_neighbor":
        match = re.search(r"(?<![a-z0-9])p(\d+)(?![a-z0-9])", normalized)
        return f"p{match.group(1)}" if match else None
    if task == "distance_constraint_audit":
        match = re.search(r"(?<![a-z0-9])c(\d+)(?![a-z0-9])", normalized)
        if match:
            return f"c{match.group(1)}"
        if re.search(r"(?<![a-z])none(?![a-z])", normalized):
            return "none"
    return None


def score_prediction(row: dict[str, Any], text: str) -> dict[str, Any]:
    target = row["target"]
    answer_type = target["answer_type"]
    if answer_type == "number":
        matches = re.findall(
            r"(?<![A-Za-z0-9_.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)",
            text,
        )
        predicted = float(matches[-1]) if matches else None
        tolerance = float(target["absolute_tolerance"])
        correct = (
            predicted is not None
            and math.isfinite(predicted)
            and abs(predicted - float(target["answer"])) <= tolerance
        )
        return {
            "prediction": predicted,
            "target": float(target["answer"]),
            "absolute_tolerance": tolerance,
            "correct": bool(correct),
        }
    predicted = extract_categorical(text, row["task"])
    expected = normalize_text(str(target["answer"])).replace(
        "no contact", "no_contact"
    )
    return {
        "prediction": predicted,
        "target": expected,
        "correct": predicted == expected,
    }


def render_prompts(
    tokenizer: Any,
    rows: list[dict[str, Any]],
) -> list[str]:
    prompts = []
    for row in rows:
        if is_ministral3_base_tokenizer(tokenizer):
            prompts.append(
                render_ministral3_base_completion(
                    SYSTEM_PROMPT, row["prompt"]
                )
            )
            continue
        if is_mllama_base_tokenizer(tokenizer):
            prompts.append(
                render_mllama_base_completion(
                    SYSTEM_PROMPT, row["prompt"]
                )
            )
            continue
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt"]},
        ]
        try:
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        except (TypeError, ValueError) as error:
            if "chat_template is not set" not in str(error) and not isinstance(error, TypeError):
                raise
            if has_gemma4_turn_tokens(tokenizer):
                raise RuntimeError(
                    "Gemma4 tokenizer is missing chat_template; install the official "
                    "instruction-tuned tokenizer assets before FTB evaluation."
                ) from error
            try:
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            except ValueError as second_error:
                if "chat_template is not set" not in str(second_error):
                    raise
                prompts.append(
                    f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{row['prompt']}\n\nAssistant:\n"
                )
    return prompts


@torch.inference_mode()
def run_mode(
    model: PeftModel,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    mode: str,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    rank: int,
) -> tuple[list[dict[str, Any]], float]:
    results = []
    started = time.perf_counter()
    context = model.disable_adapter() if mode == "base" else __import__(
        "contextlib"
    ).nullcontext()
    with context:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            prompts = render_prompts(tokenizer, batch_rows)
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(device)
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=generation_eos_token_ids(tokenizer),
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_width = encoded.input_ids.shape[1]
            decoded = tokenizer.batch_decode(
                generated[:, prompt_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for row, text in zip(batch_rows, decoded):
                results.append(
                    {
                        "id": row["id"],
                        "split": row["split"],
                        "task": row["task"],
                        "family": row["family"],
                        "answer_type": row["target"]["answer_type"],
                        "raw_output": text,
                        **score_prediction(row, text),
                    }
                )
            if rank == 0 and (
                start == 0
                or (start // batch_size + 1) % 10 == 0
                or start + batch_size >= len(rows)
            ):
                print(
                    f"[ftb:{mode}] rank0 "
                    f"{min(start + batch_size, len(rows))}/{len(rows)}",
                    flush=True,
                )
    return results, time.perf_counter() - started


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    def group_metrics(key: str) -> dict[str, Any]:
        grouped: dict[str, list[bool]] = defaultdict(list)
        for row in results:
            grouped[row[key]].append(bool(row["correct"]))
        return {
            name: {
                "accuracy": float(np.mean(values)),
                "correct": int(sum(values)),
                "examples": len(values),
            }
            for name, values in sorted(grouped.items())
        }

    return {
        "accuracy": float(
            np.mean([bool(row["correct"]) for row in results])
        ),
        "correct": int(sum(bool(row["correct"]) for row in results)),
        "examples": len(results),
        "by_split": group_metrics("split"),
        "by_family": group_metrics("family"),
        "by_task": group_metrics("task"),
    }


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError(f"Expected four GPUs, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    torch.manual_seed(20260728 + rank)

    all_rows = load_benchmark(
        args.benchmark_root, args.examples_per_task, args.selection_manifest
    )
    local_rows = all_rows[rank::world_size]
    tokenizer = load_model_level_tokenizer(args.model)
    tokenizer.padding_side = "left"
    model = load_model(args.model, args.adapter, device)
    args.output.mkdir(parents=True, exist_ok=True)

    timings = {}
    modes = ("lora",) if args.evaluation_mode == "lora-only" else ("base", "lora")
    for mode in modes:
        if mode == "base" and args.base_shards_from is not None:
            source = args.base_shards_from / f"base-rank-{rank}.jsonl"
            with source.open() as handle:
                results = [
                    json.loads(line) for line in handle if line.strip()
                ]
            expected_ids = {row["id"] for row in local_rows}
            observed_ids = {row["id"] for row in results}
            if expected_ids != observed_ids:
                raise ValueError(
                    f"Reused FTB base shard IDs do not match on rank {rank}"
                )
            elapsed = 0.0
        elif mode == "lora" and args.lora_shards_from is not None:
            source = args.lora_shards_from / f"lora-rank-{rank}.jsonl"
            with source.open() as handle:
                results = [
                    json.loads(line) for line in handle if line.strip()
                ]
            expected_ids = {row["id"] for row in local_rows}
            observed_ids = {row["id"] for row in results}
            if expected_ids != observed_ids:
                raise ValueError(
                    f"Reused FTB LoRA shard IDs do not match on rank {rank}"
                )
            elapsed = 0.0
        else:
            results, elapsed = run_mode(
                model,
                tokenizer,
                local_rows,
                mode,
                args.batch_size,
                args.max_new_tokens,
                device,
                rank,
            )
        timings[mode] = elapsed
        path = args.output / f"{mode}-rank-{rank}.jsonl"
        with path.open("w") as handle:
            for result in results:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
    dist.barrier()

    if args.evaluation_mode == "lora-only":
        dist.destroy_process_group()
        return

    timing_tensor = torch.tensor(
        [timings["base"], timings["lora"]],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(timing_tensor, op=dist.ReduceOp.MAX)
    if rank == 0:
        mode_results = {}
        for mode in ("base", "lora"):
            results = []
            for shard_rank in range(world_size):
                with (
                    args.output / f"{mode}-rank-{shard_rank}.jsonl"
                ).open() as handle:
                    results.extend(json.loads(line) for line in handle)
            results.sort(key=lambda row: row["id"])
            mode_results[mode] = aggregate(results)
        base = mode_results["base"]
        lora = mode_results["lora"]
        deltas = {
            "accuracy": lora["accuracy"] - base["accuracy"],
            "by_split": {
                split: (
                    lora["by_split"][split]["accuracy"]
                    - base["by_split"][split]["accuracy"]
                )
                for split in base["by_split"]
            },
            "by_family": {
                family: (
                    lora["by_family"][family]["accuracy"]
                    - base["by_family"][family]["accuracy"]
                )
                for family in base["by_family"]
            },
            "by_task": {
                task: (
                    lora["by_task"][task]["accuracy"]
                    - base["by_task"][task]["accuracy"]
                )
                for task in base["by_task"]
            },
        }
        summary = {
            "benchmark": "FTB-Core v1",
            "selection": (
                f"first {args.examples_per_task} stable IDs per task "
                "from each of test and test_ood"
            ),
            "examples_per_model": len(all_rows),
            "decoding": {
                "greedy": True,
                "max_new_tokens": args.max_new_tokens,
                "answer_only_system_prompt": SYSTEM_PROMPT,
            },
            "wall_seconds_max_rank": {
                "base": float(timing_tensor[0]),
                "lora": float(timing_tensor[1]),
            },
            "base": base,
            "lora": lora,
            "delta_lora_minus_base": deltas,
        }
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
