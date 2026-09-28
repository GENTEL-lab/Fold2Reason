#!/usr/bin/env python3
"""Paired base-vs-LoRA evaluation over local General-benchs text datasets."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet
import torch
import torch.distributed as dist
from peft import PeftModel
from fold2reason.models.partial_adapter import load_partial_delta
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

PROJECT = Path.cwd()

from fold2reason.analysis.paired import paired_summary  # noqa: E402


SYSTEM = (
    "You are a deterministic benchmark answer function. Do not explain, "
    "reason, restate the question, or add prose. Emit exactly the requested "
    "answer and stop."
)

TEXT_DATASETS = ["bbh", "chembench", "chembench4k", "graphqa_easy", "graphqa_hard", "lab_bench", "scibench"]

NON_TEXT_DATASETS = {
    "molecule3d_scaffold_eval": (
        "Molecular 3D geometry/property parquet. No official text-only "
        "Qwen QA adapter was defined for this run."
    ),
    "qm9": (
        "Graph/coordinate molecular property parquet. No official text-only "
        "Qwen QA adapter was defined for this run."
    ),
    "posebusters": (
        "Docking pose structure archive. Requires a pose-generation and "
        "PoseBusters physical-validity stack, not text generation."
    ),
}


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


def normalize_text(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value).strip()).lower()
    value = re.sub(
        r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*", "", value
    )
    return value.strip(" \n\t`\"'.:;")


def normalize_strict(value: str) -> str:
    return normalize_text(value).replace(" ", "")


def first_assistant_turn(value: str) -> str:
    """Discard model-generated role continuations after the first answer turn."""
    text = str(value)
    boundaries = [
        match.start()
        for match in re.finditer(r"\n\s*(?:User|System):", text)
    ]
    return text[: min(boundaries)].strip() if boundaries else text.strip()


def compact_graphqa_answer(value: str) -> str:
    return re.sub(r"[\s,;\[\]\(\)]+", "", normalize_text(value))


def graphqa_answer_tail(value: str) -> str:
    text = str(value)
    answer = None
    for pattern in (
        r"(?i)(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*",
        r"(?i)therefore,?\s*(?:the\s+answer\s+is\s*)?",
        r"(?i)so,?\s*(?:the\s+answer\s+is\s*)?",
    ):
        matches = list(re.finditer(pattern, text))
        if matches:
            answer = text[matches[-1].end() :]
            break
    if answer is None:
        answer = text
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    return lines[-1] if lines else answer


def graphqa_answer_is_correct(output: str, target: str) -> bool:
    """Paper-standard GraphQA matcher used by the canonical rescoring pass."""
    if (
        normalize_text(output) == normalize_text(target)
        or compact_graphqa_answer(output) == compact_graphqa_answer(target)
    ):
        return True
    tail = graphqa_answer_tail(output)
    if (
        normalize_text(tail) == normalize_text(target)
        or compact_graphqa_answer(tail) == compact_graphqa_answer(target)
    ):
        return True
    target_numbers = re.findall(r"-?\d+", str(target))
    if target_numbers:
        if re.findall(r"-?\d+", tail) == target_numbers:
            return True
        output_numbers = re.findall(r"-?\d+", output)
        if (
            len(output_numbers) >= len(target_numbers)
            and output_numbers[-len(target_numbers) :] == target_numbers
        ):
            return True
    target_labels = re.findall(
        r"\b(yes|no|true|false|unknown)\b", str(target).lower()
    )
    output_labels = re.findall(
        r"\b(yes|no|true|false|unknown)\b", tail.lower()
    )
    return bool(target_labels and output_labels and output_labels[-1] == target_labels[-1])


def extract_letter(value: str, choices: list[str]) -> str | None:
    max_letter = chr(ord("A") + len(choices) - 1) if choices else "Z"
    text = str(value).strip()
    patterns = [
        rf"\(([A-{max_letter}])\)",
        rf"(?<![A-Z])([A-{max_letter}])(?![A-Z])",
    ]
    upper = text.upper()
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    normalized = normalize_text(text)
    for index, choice in enumerate(choices):
        if normalize_text(choice) == normalized:
            return chr(ord("A") + index)
    return None


def extract_last_number(value: str) -> float | None:
    matches = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        str(value).replace(",", "").replace("−", "-"),
    )
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def deterministic_shuffle(values: list[str], key: str) -> list[str]:
    values = list(values)
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    random.Random(seed).shuffle(values)
    return values


def answer_letter(choices: list[str], answer: str) -> str:
    target = normalize_text(answer)
    for index, choice in enumerate(choices):
        if normalize_text(choice) == target:
            return chr(ord("A") + index)
    raise ValueError(f"Answer not found in choices: {answer}")


def option_prompt(question: str, choices: list[str]) -> str:
    lines = [question.strip(), "", "Options:"]
    for index, choice in enumerate(choices):
        lines.append(f"{chr(ord('A') + index)}. {choice}")
    lines.append("")
    lines.append("Return only the option letter.")
    return "\n".join(lines)


def score_row(row: dict[str, Any], output: str) -> dict[str, Any]:
    output = first_assistant_turn(output)
    if row["scoring"] == "graphqa_canonical":
        target = str(row["target"])
        correct = graphqa_answer_is_correct(output, target)
        return {
            "prediction": normalize_text(output)[:256],
            "target": normalize_text(target)[:256],
            "score": 1.0 if correct else 0.0,
            "correct": bool(correct),
        }
    scoring = row["scoring"]
    if scoring == "multiple_choice":
        prediction = extract_letter(output, row["choices"])
        correct = prediction == row["target"]
        return {
            "prediction": prediction,
            "target": row["target"],
            "score": 1.0 if correct else 0.0,
            "correct": bool(correct),
        }
    if scoring == "numeric_1pct":
        prediction = extract_last_number(output)
        target = float(row["target"])
        correct = (
            prediction is not None
            and math.isfinite(prediction)
            and math.isclose(
                prediction,
                target,
                rel_tol=0.01,
                abs_tol=max(1e-4, abs(target) * 1e-4),
            )
        )
        return {
            "prediction": prediction,
            "target": target,
            "score": 1.0 if correct else 0.0,
            "correct": bool(correct),
        }
    if scoring == "normalized_exact":
        prediction = normalize_text(output)
        target = normalize_text(row["target"])
        correct = prediction == target
        return {
            "prediction": prediction,
            "target": target,
            "score": 1.0 if correct else 0.0,
            "correct": bool(correct),
        }
    if scoring == "strict_compact_exact":
        prediction = normalize_strict(output)
        target = normalize_strict(row["target"])
        correct = prediction == target
        return {
            "prediction": prediction[:256],
            "target": target[:256],
            "score": 1.0 if correct else 0.0,
            "correct": bool(correct),
        }
    if scoring == "vsi_mixed":
        target_number = extract_last_number(row["target"])
        prediction_number = extract_last_number(output)
        if target_number is not None:
            if prediction_number is None or not math.isfinite(
                prediction_number
            ):
                relative = 0.0
            else:
                error = abs(prediction_number - target_number)
                relative = max(
                    0.0, 1.0 - error / max(abs(target_number), 1e-6)
                )
            return {
                "prediction": prediction_number,
                "target": target_number,
                "score": float(relative),
                "correct": bool(relative >= 0.9),
            }
        prediction = normalize_text(output)
        target = normalize_text(row["target"])
        correct = prediction == target
        return {
            "prediction": prediction,
            "target": target,
            "score": 1.0 if correct else 0.0,
            "correct": bool(correct),
        }
    raise ValueError(f"Unknown scoring: {scoring}")


def stable_select(
    rows: list[dict[str, Any]],
    max_examples: int,
) -> list[dict[str, Any]]:
    if len(rows) <= max_examples:
        return sorted(rows, key=lambda row: row["id"])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: row["id"])
    selected = []
    position = 0
    while len(selected) < max_examples:
        added = False
        for key in sorted(groups):
            if position < len(groups[key]):
                selected.append(groups[key][position])
                added = True
                if len(selected) == max_examples:
                    break
        if not added:
            break
        position += 1
    return sorted(selected, key=lambda row: row["id"])


def read_parquet(path: Path, columns: list[str] | None = None) -> list[dict]:
    return parquet.read_table(path, columns=columns).to_pylist()


def load_bbh(root: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    available = 0
    for path in sorted(root.glob("*/test-*.parquet")):
        task = path.parent.name
        table = read_parquet(path)
        available += len(table)
        for index, item in enumerate(table):
            target = str(item["target"]).strip()
            scoring = (
                "multiple_choice"
                if re.fullmatch(r"\([A-Z]\)", target)
                else "normalized_exact"
            )
            choices = []
            target_letter = target.strip("()") if scoring == "multiple_choice" else target
            rows.append(
                {
                    "id": f"bbh:{task}:{index:05d}",
                    "dataset": "bbh",
                    "category": task,
                    "prompt": item["input"],
                    "target": target_letter,
                    "choices": choices,
                    "scoring": scoring,
                }
            )
    return rows, available


def load_chembench(root: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    available = 0
    for path in sorted(root.glob("*/train-*.parquet")):
        category = path.parent.name
        table = read_parquet(path)
        for item_index, item in enumerate(table):
            for ex_index, example in enumerate(item.get("examples") or []):
                if not example.get("target_scores"):
                    continue
                scores = json.loads(example["target_scores"])
                best_score = max(float(value) for value in scores.values())
                correct = [
                    key
                    for key, value in scores.items()
                    if float(value) == best_score
                ]
                if len(correct) != 1:
                    continue
                choices = deterministic_shuffle(
                    list(scores), f"chembench:{category}:{item_index}:{ex_index}"
                )
                rows.append(
                    {
                        "id": (
                            f"chembench:{category}:{item_index:05d}:"
                            f"{ex_index:02d}"
                        ),
                        "dataset": "chembench",
                        "category": category,
                        "prompt": option_prompt(example["input"], choices),
                        "target": answer_letter(choices, correct[0]),
                        "choices": choices,
                        "scoring": "multiple_choice",
                    }
                )
                available += 1
    return rows, available


def load_chembench4k(root: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    available = 0
    for path in sorted((root / "test").glob("*_benchmark.json")):
        category = path.stem.replace("_benchmark", "")
        items = json.loads(path.read_text())
        available += len(items)
        for index, item in enumerate(items):
            choices = [str(item[letter]) for letter in ("A", "B", "C", "D")]
            rows.append(
                {
                    "id": f"chembench4k:{category}:{index:05d}",
                    "dataset": "chembench4k",
                    "category": category,
                    "prompt": option_prompt(item["question"], choices),
                    "target": str(item["answer"]).strip().upper(),
                    "choices": choices,
                    "scoring": "multiple_choice",
                }
            )
    return rows, available


def load_clrs(root: Path, max_examples: int) -> tuple[list[dict[str, Any]], int]:
    rows = []
    available = 0
    per_file = max(1, math.ceil(max_examples / 5))
    for path in sorted((root / "data").glob("test_*-*.parquet")):
        pf = parquet.ParquetFile(path)
        available += pf.metadata.num_rows
        collected = 0
        for batch in pf.iter_batches(
            batch_size=512,
            columns=["question", "answer", "algo_name", "length"],
        ):
            for item in batch.to_pylist():
                rows.append(
                    {
                        "id": (
                            f"clrs_text_test:{path.stem}:"
                            f"{collected:06d}"
                        ),
                        "dataset": "clrs_text_test",
                        "category": str(item["algo_name"]),
                        "prompt": item["question"],
                        "target": item["answer"],
                        "choices": [],
                        "scoring": "strict_compact_exact",
                    }
                )
                collected += 1
                if collected >= per_file:
                    break
            if collected >= per_file:
                break
    return rows, available


def load_graphqa(root: Path, dataset: str) -> tuple[list[dict[str, Any]], int]:
    path = root / "data" / "test-00000-of-00001.parquet"
    table = read_parquet(path)
    rows = []
    for index, item in enumerate(table):
        rows.append(
            {
                "id": f"{dataset}:{item['task']}:{index:06d}",
                "dataset": dataset,
                "category": str(item["task"]),
                "prompt": item["question"],
                "target": item["answer"],
                "choices": [],
                "scoring": "graphqa_canonical",
            }
        )
    return rows, len(table)


def load_lab_bench(root: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    available = 0
    for path in sorted(root.glob("*/train-*.parquet")):
        category = path.parent.name
        table = read_parquet(path)
        available += len(table)
        for index, item in enumerate(table):
            if "question" not in item or "ideal" not in item:
                continue
            distractors = [str(value) for value in item.get("distractors") or []]
            if distractors:
                choices = deterministic_shuffle(
                    [str(item["ideal"])] + distractors,
                    f"lab_bench:{category}:{index}",
                )
                prompt = option_prompt(str(item["question"]), choices)
                target = answer_letter(choices, str(item["ideal"]))
                scoring = "multiple_choice"
            else:
                choices = []
                prompt = str(item["question"])
                target = str(item["ideal"])
                scoring = "normalized_exact"
            rows.append(
                {
                    "id": f"lab_bench:{category}:{index:05d}",
                    "dataset": "lab_bench",
                    "category": category,
                    "prompt": prompt,
                    "target": target,
                    "choices": choices,
                    "scoring": scoring,
                }
            )
    return rows, available


def load_planbench(root: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    available = 0
    for path in sorted(root.glob("*/train-*.parquet")):
        table = read_parquet(path)
        available += len(table)
        for index, item in enumerate(table):
            rows.append(
                {
                    "id": f"planbench:{item['task']}:{index:05d}",
                    "dataset": "planbench",
                    "category": str(item["task"]),
                    "prompt": item["query"],
                    "target": item["ground_truth_plan"],
                    "choices": [],
                    "scoring": "strict_compact_exact",
                }
            )
    return rows, available


def parse_float(value: str) -> float:
    return float(
        str(value)
        .strip()
        .replace(",", "")
        .replace("−", "-")
        .replace("\\times", "e")
    )


def load_scibench(root: Path) -> tuple[list[dict[str, Any]], int]:
    rows = []
    available = 0
    for path in sorted(root.glob("*.json")):
        if path.name == "SOURCE.json" or path.stem.endswith("_sol"):
            continue
        items = json.loads(path.read_text())
        available += len(items)
        for index, item in enumerate(items):
            rows.append(
                {
                    "id": f"scibench:{path.stem}:{index:05d}",
                    "dataset": "scibench",
                    "category": path.stem,
                    "prompt": item["problem_text"],
                    "target": parse_float(item["answer_number"]),
                    "choices": [],
                    "scoring": "numeric_1pct",
                }
            )
    return rows, available


def load_spatialviz(root: Path) -> tuple[list[dict[str, Any]], int]:
    path = root / "data" / "test-00000-of-00001.parquet"
    table = read_parquet(path)
    rows = []
    for index, item in enumerate(table):
        choices = [str(value) for value in item["Choices"]]
        rows.append(
            {
                "id": f"spatialviz_bench_text_only:{index:05d}",
                "dataset": "spatialviz_bench_text_only",
                "category": f"{item['Category']}:{item['Task']}",
                "prompt": option_prompt(item["Question"], choices),
                "target": str(item["Answer"]).strip().upper(),
                "choices": choices,
                "scoring": "multiple_choice",
            }
        )
    return rows, len(table)


def load_vsi(root: Path) -> tuple[list[dict[str, Any]], int]:
    path = root / "test_debiased.parquet"
    table = read_parquet(path)
    rows = []
    for item in table:
        options = item.get("options")
        choices = (
            [str(value) for value in options]
            if isinstance(options, list)
            else []
        )
        if choices:
            prompt = option_prompt(item["question"], choices)
            scoring = "multiple_choice"
            target = str(item["ground_truth"]).strip().upper()
        else:
            prompt = item["question"]
            scoring = "vsi_mixed"
            target = str(item["ground_truth"]).strip()
        rows.append(
            {
                "id": f"vsi_bench_text_only:{int(item['id']):05d}",
                "dataset": "vsi_bench_text_only",
                "category": str(item["question_type"]),
                "prompt": prompt,
                "target": target,
                "choices": choices,
                "scoring": scoring,
            }
        )
    return rows, len(table)


def load_ftb_counts(root: Path) -> dict[str, int]:
    counts = {}
    for split in ("train", "validation", "test", "test_ood"):
        path = root / f"{split}.jsonl.gz"
        if not path.is_file():
            continue
        with gzip.open(path, "rt") as handle:
            counts[split] = sum(1 for _ in handle)
    return counts


def load_all_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = args.general_benchs_root / "data" / "external"
    loaders = {
        "bbh": lambda: load_bbh(root / "bbh"),
        "chembench": lambda: load_chembench(root / "chembench"),
        "chembench4k": lambda: load_chembench4k(root / "chembench4k"),
        "clrs_text_test": lambda: load_clrs(
            root / "clrs_text_test", args.max_examples_per_dataset
        ),
        "graphqa_easy": lambda: load_graphqa(
            root / "graphqa_easy", "graphqa_easy"
        ),
        "graphqa_hard": lambda: load_graphqa(
            root / "graphqa_hard", "graphqa_hard"
        ),
        "lab_bench": lambda: load_lab_bench(root / "lab_bench"),
        "planbench": lambda: load_planbench(root / "planbench"),
        "scibench": lambda: load_scibench(root / "scibench"),
        "spatialviz_bench_text_only": lambda: load_spatialviz(
            root / "spatialviz_bench"
        ),
        "vsi_bench_text_only": lambda: load_vsi(root / "vsi_bench"),
    }
    selected = []
    manifest: dict[str, Any] = {}
    for dataset in args.datasets:
        rows, available = loaders[dataset]()
        if dataset in {"graphqa_easy", "graphqa_hard"}:
            scoring = (
                "graphqa_canonical"
                if args.graphqa_scoring == "canonical"
                else "normalized_exact"
            )
            for row in rows:
                row["scoring"] = scoring
        use_rows = stable_select(rows, args.max_examples_per_dataset)
        manifest[dataset] = {
            "available_examples": available,
            "selected_examples": len(use_rows),
            "categories": len({row["category"] for row in rows}),
            "scoring_modes": sorted({row["scoring"] for row in rows}),
        }
        selected.extend(use_rows)
    for dataset, note in NON_TEXT_DATASETS.items():
        manifest[dataset] = {
            "available_examples": None,
            "selected_examples": 0,
            "categories": None,
            "scoring_modes": [],
            "not_evaluated_reason": note,
        }
    ftb_root = args.general_benchs_root / "data/generated/ftb_core/v1"
    manifest["ftb_core_v1"] = {
        "available_examples_by_split": load_ftb_counts(ftb_root),
        "selected_examples": 0,
        "scoring_modes": ["strict_normalized_exact_match"],
        "note": (
            "Not regenerated by this script; the report can reuse the "
            "completed full 12k base-vs-B run."
        ),
    }
    return sorted(selected, key=lambda row: row["id"]), manifest


def load_base_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    if local_model_type(model_path) == "internvl_chat":
        return load_internvl35_text_model(model_path, device)
    if local_model_type(model_path) == "mistral3":
        return load_ministral3_text_model(model_path, device)
    raw_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    text_config = getattr(raw_config, "text_config", None)
    if getattr(text_config, "model_type", None) == "qwen3_5":
        text_config._attn_implementation = "sdpa"
        model = Qwen3_5ForCausalLM.from_pretrained(
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
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=raw_config,
            dtype=torch.bfloat16,
            device_map={"": device.index},
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    model.eval()
    return model


def render_chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    if is_ministral3_base_tokenizer(tokenizer):
        return render_ministral3_base_completion(SYSTEM, user_prompt)
    if is_mllama_base_tokenizer(tokenizer):
        return render_mllama_base_completion(SYSTEM, user_prompt)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError) as error:
        if "chat_template is not set" not in str(error) and not isinstance(error, TypeError):
            raise
        if has_gemma4_turn_tokens(tokenizer):
            raise RuntimeError(
                "Gemma4 tokenizer is missing chat_template; install the official "
                "chat_template.jinja and rerender/re-evaluate instead of using the "
                "plain System/User/Assistant fallback."
            ) from error
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except ValueError as second_error:
            if "chat_template is not set" not in str(second_error):
                raise
    return f"System:\n{SYSTEM}\n\nUser:\n{user_prompt}\n\nAssistant:\n"


def load_model(
    model_path: Path,
    adapter_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    base = load_base_model(model_path, device)
    if (adapter_path / "partial_ft_delta.pt").exists():
        model = base
        load_partial_delta(model, adapter_path)
    else:
        model = PeftModel.from_pretrained(
            base, adapter_path, is_trainable=False
        )
    model.eval()
    return model


@torch.inference_mode()
def evaluate_mode(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    mode: str,
    batch_size: int,
    max_new_tokens: int,
    planbench_max_new_tokens: int,
    max_input_tokens: int,
    use_cache: bool,
    device: torch.device,
    rank: int,
) -> tuple[list[dict[str, Any]], float]:
    results = []
    started = time.perf_counter()

    def run_batch(
        batch: list[dict[str, Any]],
        active_batch_size: int,
        active_use_cache: bool = use_cache,
    ) -> list[dict[str, Any]]:
        prompts = [render_chat_prompt(tokenizer, row["prompt"]) for row in batch]
        generation_limit = max(
            max_new_tokens,
            *(
                planbench_max_new_tokens if row["dataset"] == "planbench" else 0
                for row in batch
            ),
        )
        encoded = None
        retries = None
        try:
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
                add_special_tokens=False,
            ).to(device)
            input_lengths = encoded.attention_mask.sum(dim=1).tolist()
            if max(input_lengths) > max_input_tokens:
                offending = [
                    (row["id"], length)
                    for row, length in zip(batch, input_lengths)
                    if length > max_input_tokens
                ]
                raise RuntimeError(
                    f"Input exceeds --max-input-tokens={max_input_tokens}; "
                    f"refusing silent truncation: {offending}"
                )
            generated = model.generate(
                **encoded,
                max_new_tokens=generation_limit,
                do_sample=False,
                use_cache=active_use_cache,
                eos_token_id=generation_eos_token_ids(tokenizer),
                pad_token_id=tokenizer.pad_token_id,
            )
            width = encoded.input_ids.shape[1]
            completion_ids = generated[:, width:]
            decoded = tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output_lengths = (completion_ids != tokenizer.pad_token_id).sum(dim=1).tolist()
        except torch.cuda.OutOfMemoryError:
            del prompts
            del encoded
            if len(batch) == 1:
                if active_use_cache:
                    if rank == 0:
                        print(
                            f"[general:{mode}] OOM fallback batch=1 "
                            "use_cache=True -> False",
                            flush=True,
                        )
                    retries = [(batch, 1, False)]
                else:
                    raise
            else:
                midpoint = len(batch) // 2
                if rank == 0:
                    print(
                        f"[general:{mode}] OOM fallback "
                        f"{len(batch)} -> {midpoint}+{len(batch) - midpoint}",
                        flush=True,
                    )
                retries = [
                    (batch[:midpoint], midpoint, active_use_cache),
                    (batch[midpoint:], len(batch) - midpoint, active_use_cache),
                ]
        if retries is not None:
            torch.cuda.empty_cache()
            retried_outputs = []
            for retry_batch, retry_size, retry_use_cache in retries:
                retried_outputs.extend(
                    run_batch(retry_batch, retry_size, retry_use_cache)
                )
            return retried_outputs
        outputs = []
        for row, output, input_tokens, output_tokens in zip(
            batch, decoded, input_lengths, output_lengths
        ):
            outputs.append(
                {
                    "id": row["id"],
                    "dataset": row["dataset"],
                    "category": row["category"],
                    "scoring": row["scoring"],
                    "raw_output": output,
                    "input_tokens": int(input_tokens),
                    "output_tokens": int(output_tokens),
                    "effective_batch_size": active_batch_size,
                    "effective_use_cache": active_use_cache,
                    "generation_max_new_tokens": generation_limit,
                    **score_row(row, output),
                }
            )
        return outputs

    context = (
        model.disable_adapter()
        if mode == "base" and hasattr(model, "disable_adapter")
        else contextlib.nullcontext()
    )
    with context:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            results.extend(run_batch(batch, len(batch)))
            if rank == 0 and (
                start == 0
                or (start // batch_size + 1) % 25 == 0
                or start + batch_size >= len(rows)
            ):
                print(
                    f"[general:{mode}] rank0 "
                    f"{min(start + batch_size, len(rows))}/{len(rows)}",
                    flush=True,
                )
    return results, time.perf_counter() - started


def summarize_scores(
    base: dict[str, dict[str, Any]],
    lora: dict[str, dict[str, Any]],
    ids: list[str],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    paired = paired_summary(
        [
            (bool(base[item_id]["correct"]), bool(lora[item_id]["correct"]))
            for item_id in ids
        ],
        rng,
        bootstrap_samples,
    )
    base_scores = np.asarray([float(base[item_id]["score"]) for item_id in ids])
    lora_scores = np.asarray([float(lora[item_id]["score"]) for item_id in ids])
    paired.update(
        {
            "base_mean_score": float(base_scores.mean()),
            "lora_mean_score": float(lora_scores.mean()),
            "delta_mean_score": float(lora_scores.mean() - base_scores.mean()),
        }
    )
    return paired


def write_summary(
    output: Path,
    manifest: dict[str, Any],
    args: argparse.Namespace,
    world_size: int,
    timings: torch.Tensor,
) -> None:
    modes = {}
    for mode in ("base", "lora"):
        mode_rows = []
        for rank in range(world_size):
            with (output / f"{mode}-rank-{rank}.jsonl").open() as handle:
                mode_rows.extend(json.loads(line) for line in handle)
        modes[mode] = {row["id"]: row for row in mode_rows}
    if set(modes["base"]) != set(modes["lora"]):
        raise ValueError("Base and LoRA result ids do not match")
    rng = np.random.default_rng(args.seed)
    ids = sorted(modes["base"])
    summary = {
        "experiment": "general_benchs_all_text_eval",
        "seed": args.seed,
        "model": str(args.model),
        "adapter": str(args.adapter),
        "general_benchs_root": str(args.general_benchs_root),
        "system_prompt": SYSTEM,
        "prompt_protocol": {
            "renderer": "tokenizer.apply_chat_template",
            "add_generation_prompt": True,
            "enable_thinking": False,
            "response_boundary": "first assistant turn before a generated User/System role",
            "graphqa_matcher": (
                "paper canonical normalized/compact/final-answer matcher"
                if args.graphqa_scoring == "canonical"
                else "historical normalized exact matcher"
            ),
        },
        "decoding": {
            "greedy": True,
            "max_new_tokens": args.max_new_tokens,
            "planbench_max_new_tokens": args.planbench_max_new_tokens,
            "batch_size_per_rank": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "max_observed_input_tokens": max(
                int(row["input_tokens"]) for row in modes["lora"].values()
            ),
            "truncated_inputs": 0,
            "no_cache_fallback_examples": sum(
                not bool(row.get("effective_use_cache", True))
                for row in modes["lora"].values()
            ),
            "use_cache": not args.no_use_cache,
        },
        "selection": {
            "policy": (
                "deterministic round-robin by category, capped per dataset; "
                "small datasets use all available text-evaluable examples"
            ),
            "max_examples_per_dataset": args.max_examples_per_dataset,
        },
        "dataset_manifest": manifest,
        "overall_text_evaluable": summarize_scores(
            modes["base"],
            modes["lora"],
            ids,
            rng,
            args.bootstrap_samples,
        ),
        "by_dataset": {},
        "by_category": {},
        "wall_seconds_max_rank": {
            "base": float(timings[0].item()),
            "lora": float(timings[1].item()),
        },
    }
    by_dataset: dict[str, list[str]] = defaultdict(list)
    by_category: dict[str, list[str]] = defaultdict(list)
    for item_id in ids:
        row = modes["base"][item_id]
        by_dataset[row["dataset"]].append(item_id)
        by_category[f"{row['dataset']}:{row['category']}"].append(item_id)
    for dataset, group_ids in sorted(by_dataset.items()):
        summary["by_dataset"][dataset] = summarize_scores(
            modes["base"],
            modes["lora"],
            sorted(group_ids),
            rng,
            args.bootstrap_samples,
        )
    for category, group_ids in sorted(by_category.items()):
        summary["by_category"][category] = summarize_scores(
            modes["base"],
            modes["lora"],
            sorted(group_ids),
            rng,
            max(500, args.bootstrap_samples // 5),
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3.5-9B"),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path(
            "outputs/qwen35_9b_openfold_1k_v2/geohead/"
            "seed-20260729/checkpoints/best/adapter"
        ),
    )
    parser.add_argument(
        "--general-benchs-root",
        type=Path,
        default=Path("data/benchmarks"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=TEXT_DATASETS,
        default=TEXT_DATASETS,
    )
    parser.add_argument("--max-examples-per-dataset", type=int, default=0)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=None,
        help="Exact general-mini-manifest-v1 IDs; overrides the per-dataset cap.",
    )
    parser.add_argument(
        "--data-shard-count",
        type=int,
        default=1,
        help=(
            "Split the globally sorted selected rows into this many disjoint "
            "strided shards. This is independent of the four local GPU ranks."
        ),
    )
    parser.add_argument(
        "--data-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index used with --data-shard-count.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--planbench-max-new-tokens",
        type=int,
        default=1280,
        help="PlanBench generation cap; use 128 to reproduce the historical run.",
    )
    parser.add_argument(
        "--graphqa-scoring",
        choices=("canonical", "normalized_exact"),
        default="canonical",
        help="GraphQA scoring protocol; normalized_exact reproduces the historical run.",
    )
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument(
        "--no-use-cache",
        action="store_true",
        help="Disable KV cache during generation to reduce peak memory on long prompts.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--base-shards-from",
        type=Path,
        help=(
            "Reuse base-rank-*.jsonl from an identical prior selection. "
            "IDs are checked against this run before use."
        ),
    )
    parser.add_argument(
        "--lora-shards-from",
        type=Path,
        help=(
            "Reuse lora-rank-*.jsonl from an identical prior selection. "
            "IDs are checked against this run before use."
        ),
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("both", "lora-only"),
        default="both",
        help=(
            "lora-only writes audited adapter shards without a summary so "
            "adapter inference can run before compatible Base shards arrive."
        ),
    )
    parser.add_argument(
        "--allow-partial-base-reload",
        action="store_true",
        help=(
            "For partial/full-FT artifacts, load a second frozen base tower "
            "and evaluate it directly when reusable base shards do not exist."
        ),
    )
    parser.add_argument(
        "--exclude-ids-from",
        type=Path,
        default=None,
        help=(
            "Exclude item IDs found in base-rank-*.jsonl under this directory. "
            "Used to keep the confirmatory complement sealed from recipe-selection dev items."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/qwen35_9b_openfold_1k_v2/geohead/"
            "seed-20260729/general_benchs_all"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        (args.adapter / "partial_ft_delta.pt").exists()
        and args.base_shards_from is None
        and not args.allow_partial_base_reload
    ):
        raise ValueError(
            "partial/full-FT General evaluation requires --base-shards-from "
            "or --allow-partial-base-reload"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size not in (4, 8):
        raise RuntimeError(f"Expected four or eight GPUs, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=device,
        timeout=timedelta(hours=24),
    )
    if args.data_shard_count < 1:
        raise ValueError("--data-shard-count must be positive")
    if args.planbench_max_new_tokens < 1:
        raise ValueError("--planbench-max-new-tokens must be positive")
    if not 0 <= args.data_shard_index < args.data_shard_count:
        raise ValueError(
            "--data-shard-index must be in [0, --data-shard-count)"
        )
    configured_max_examples = args.max_examples_per_dataset
    fixed_selection = None
    if args.selection_manifest is not None:
        fixed_selection = json.loads(args.selection_manifest.read_text())
        args.max_examples_per_dataset = 1_000_000_000
    all_rows, manifest = load_all_rows(args)
    args.max_examples_per_dataset = configured_max_examples
    if fixed_selection is not None:
        if fixed_selection.get("format") != "general-mini-manifest-v1":
            raise ValueError("--selection-manifest must use general-mini-manifest-v1")
        requested = {
            str(item_id)
            for dataset in args.datasets
            for item_id in fixed_selection["datasets"][dataset]["ids"]
        }
        available = {row["id"]: row for row in all_rows}
        missing = sorted(requested - set(available))
        if missing:
            raise ValueError(
                f"selection manifest contains {len(missing)} unavailable IDs; "
                f"first={missing[:3]}"
            )
        all_rows = sorted(
            (available[item_id] for item_id in requested), key=lambda row: row["id"]
        )
        selected_ids = [row["id"] for row in all_rows]
        observed_hash = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
        if observed_hash != fixed_selection["global_id_sha256"]:
            raise ValueError("selection manifest ID hash mismatch")
        manifest["_fixed_selection_manifest"] = {
            "path": str(args.selection_manifest),
            "format": fixed_selection["format"],
            "seed": fixed_selection["seed"],
            "selected_examples": len(all_rows),
            "global_id_sha256": observed_hash,
        }
    if args.exclude_ids_from is not None:
        exclude_ids: set[str] = set()
        source_files = sorted(args.exclude_ids_from.glob("base-rank-*.jsonl"))
        if not source_files:
            raise FileNotFoundError(
                f"no base-rank shards found under {args.exclude_ids_from}"
            )
        for source in source_files:
            with source.open() as handle:
                for line in handle:
                    exclude_ids.add(json.loads(line)["id"])
        before = len(all_rows)
        all_rows = [row for row in all_rows if row["id"] not in exclude_ids]
        manifest["_excluded_recipe_dev"] = {
            "source": str(args.exclude_ids_from),
            "source_files": [str(path) for path in source_files],
            "excluded_ids": before - len(all_rows),
            "requested_exclude_ids": len(exclude_ids),
            "exclude_id_sha256": hashlib.sha256(
                "\n".join(sorted(exclude_ids)).encode()
            ).hexdigest(),
        }
    global_ids = [row["id"] for row in all_rows]
    rows = all_rows[args.data_shard_index :: args.data_shard_count]
    manifest["_distributed_data_shard"] = {
        "policy": "globally sorted row ids, strided by shard count",
        "count": args.data_shard_count,
        "index": args.data_shard_index,
        "global_selected_examples": len(all_rows),
        "shard_selected_examples": len(rows),
        "global_id_sha256": hashlib.sha256(
            "\n".join(global_ids).encode()
        ).hexdigest(),
        "shard_id_sha256": hashlib.sha256(
            "\n".join(row["id"] for row in rows).encode()
        ).hexdigest(),
    }
    local_rows = rows[rank::world_size]
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "selected_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"[general] selected={len(rows)} "
            f"rank0_rows={len(local_rows)} output={args.output}",
            flush=True,
        )
    dist.barrier()
    tokenizer = load_model_level_tokenizer(args.model)
    tokenizer.padding_side = "left"
    model = load_model(args.model, args.adapter, device)
    direct_base_model = None
    if (
        (args.adapter / "partial_ft_delta.pt").exists()
        and args.base_shards_from is None
    ):
        direct_base_model = load_base_model(args.model, device)
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
                    "Reused General-benchs base shard IDs do not match "
                    f"on rank {rank}"
                )
            timings[mode] = 0.0
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
                    "Reused General-benchs LoRA shard IDs do not match "
                    f"on rank {rank}"
                )
            timings[mode] = 0.0
        else:
            evaluation_model = (
                direct_base_model
                if mode == "base" and direct_base_model is not None
                else model
            )
            results, timings[mode] = evaluate_mode(
                evaluation_model,
                tokenizer,
                local_rows,
                mode,
                args.batch_size,
                args.max_new_tokens,
                args.planbench_max_new_tokens,
                args.max_input_tokens,
                not args.no_use_cache,
                device,
                rank,
            )
        with (args.output / f"{mode}-rank-{rank}.jsonl").open("w") as handle:
            for row in results:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
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
        write_summary(
            args.output, manifest, args, world_size, timing_tensor.cpu()
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
