#!/usr/bin/env python3
"""Shared data, prompting, scoring, and LoRA helpers for visual benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import re
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet
import torch
from peft import PeftConfig, inject_adapter_in_model
from safetensors.torch import load_file


SPATIALVIZ_REVISION = "f38482a83f8f29e3cf9e07cb131d4de0f2cbd2f0"
VSI_REVISION = "d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d"

VSI_MCA_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}
VSI_NUMERIC_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}
VSI_REPORTED_TASKS = (
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
    "object_rel_distance",
    "object_rel_direction",
    "route_planning",
    "obj_appearance_order",
)

ADAPTER_KEY = re.compile(
    r"^base_model\.model\.model\.(.+)\.lora_([AB])\.weight$"
)


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_spatialviz_rows(root: Path) -> list[dict[str, Any]]:
    annotation = root / "data" / "test-00000-of-00001.parquet"
    rows = parquet.read_table(annotation).to_pylist()
    output = []
    for index, row in enumerate(rows):
        image_path = (
            root
            / "SpatialViz_Bench_images"
            / str(row["Category"])
            / str(row["Task"])
            / str(row["Level"])
            / f"{row['Image_id']}.png"
        )
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        choices = [str(value) for value in row["Choices"]]
        if len(choices) != 4:
            raise ValueError(f"SpatialViz row {index} has {len(choices)} choices")
        output.append(
            {
                "id": f"spatialviz_multimodal:{index:05d}",
                "index": index,
                "category": str(row["Category"]),
                "task": str(row["Task"]),
                "level": str(row["Level"]),
                "image_id": str(row["Image_id"]),
                "image_path": str(image_path),
                "question": str(row["Question"]),
                "choices": choices,
                "target": str(row["Answer"]).strip().upper(),
                "cache_key": f"spatialviz/{index:05d}.pt",
            }
        )
    if len(output) != 1180 or len({row["id"] for row in output}) != 1180:
        raise RuntimeError("SpatialViz must contain exactly 1,180 unique rows")
    return output


def load_vsi_rows(root: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (root / "test.jsonl").read_text().splitlines()
        if line.strip()
    ]
    debiased_ids = {
        int(row["id"])
        for row in parquet.read_table(root / "test_debiased.parquet").to_pylist()
    }
    output = []
    for row in rows:
        item_id = int(row["id"])
        question_type = str(row["question_type"])
        if question_type not in VSI_MCA_TYPES | VSI_NUMERIC_TYPES:
            raise ValueError(f"Unknown VSI question type: {question_type}")
        video_path = (
            root / str(row["dataset"]) / f"{row['scene_name']}.mp4"
        )
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        options = row.get("options")
        output.append(
            {
                "id": f"vsi_multimodal:{item_id:05d}",
                "numeric_id": item_id,
                "dataset": str(row["dataset"]),
                "scene_name": str(row["scene_name"]),
                "question_type": question_type,
                "question": str(row["question"]),
                "options": (
                    [str(value) for value in options]
                    if isinstance(options, list)
                    else []
                ),
                "target": str(row["ground_truth"]).strip(),
                "debiased": item_id in debiased_ids,
                "video_path": str(video_path),
                "cache_key": (
                    f"vsi/{row['dataset']}/{row['scene_name']}.pt"
                ),
            }
        )
    if len(output) != 5130 or len({row["id"] for row in output}) != 5130:
        raise RuntimeError("VSI-Bench must contain exactly 5,130 unique rows")
    if sum(row["debiased"] for row in output) != 2362:
        raise RuntimeError("VSI debiased subset must contain exactly 2,362 rows")
    if len({row["cache_key"] for row in output}) != 288:
        raise RuntimeError("VSI-Bench must resolve to exactly 288 scene videos")
    return output


def spatialviz_prompt(row: dict[str, Any]) -> str:
    choices = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip("ABCD", row["choices"])
    )
    return (
        "Answer with a single option letter (A, B, C, or D), enclosed within "
        "the <answer></answer> tag. For example: <answer>A</answer>. Ensure "
        "that your output contains only the final answer, without any "
        "intermediate reasoning or additional content.\n"
        f"Question: {row['question'].strip()}\n"
        f"Choices: {choices}"
    )


def vsi_prompt(row: dict[str, Any]) -> str:
    pre_prompt = "These are frames of a video."
    if row["question_type"] in VSI_NUMERIC_TYPES:
        return (
            f"{pre_prompt}\n{row['question']}\n"
            "Please answer the question using a single word or phrase."
        )
    options = "Options:\n" + "\n".join(row["options"])
    return "\n".join(
        [
            pre_prompt,
            row["question"],
            options,
            "Answer with the option's letter from the given choices directly.",
        ]
    )


def multimodal_messages(
    media_type: str,
    media_path: str,
    prompt: str,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": media_type, "path": media_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def expand_cached_media_tokens(
    processor: Any,
    messages: list[dict[str, Any]],
    media_token_id: int,
    replacement_ids: list[int],
    enable_thinking: bool = False,
) -> list[int]:
    rendered = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
    raw_ids = processor.tokenizer(
        rendered,
        add_special_tokens=False,
    ).input_ids
    positions = [
        index for index, token_id in enumerate(raw_ids)
        if token_id == media_token_id
    ]
    if len(positions) != 1:
        raise RuntimeError(
            f"Expected one media placeholder, found {len(positions)}"
        )
    index = positions[0]
    return raw_ids[:index] + replacement_ids + raw_ids[index + 1 :]


def extract_choice_letter(value: str) -> str | None:
    """Reproduce SpatialViz-Bench's strict default answer extraction."""
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 1:
        matches = re.findall(r"[A-D]", text)
        return matches[0] if len(matches) == 1 else None

    legacy_markers = (
        "<answer>",
        "Answer:",
        "Final answer",
        "final answer",
        "Final Answer",
        "the answer is",
        "The answer is",
        "correct answer",
        "Correct answer",
        "Correct Answer",
        "correct path",
    )
    for marker in legacy_markers:
        if marker not in text:
            continue
        matches = re.findall(
            r"\b([A-D])\b",
            text.split(marker)[-1].strip().split(".")[0],
        )
        if matches:
            unique = set(matches)
            return unique.pop() if len(unique) == 1 else None

    patterns = (
        r"<answer>\s*(?P<value>.*?)\s*</answer>",
        r"<answer>\s*option\s+(?P<value>[A-D])(?=answer>)",
        r"</answer>\s*(?P<value>[A-D])\b",
        (
            r"(?:final|correct\s+)?answer\s*(?:is|:)\s*"
            r"(?:option\s*)?(?P<value>[A-D])\b"
        ),
        (
            r"correct\s+path\s*(?:is|:)?\s*"
            r"(?:option\s*)?(?P<value>[A-D])\b"
        ),
        (
            r"correct\s+choice\s*(?:is|:)?\s*"
            r"(?:option\s*)?(?P<value>[A-D])\b"
        ),
        r"option\s+(?P<value>[A-D])\b",
        r"choose\s+(?P<value>[A-D])\b",
        r"\\{1,2}boxed\{(?:\\text\{)?(?P<value>[A-D])",
    )
    for pattern in patterns:
        answers = set()
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            raw = match.group("value").strip()
            values = re.findall(r"\b([A-D])\b", raw)
            if len(values) == 1:
                answers.add(values[0].upper())
        if answers:
            return answers.pop() if len(answers) == 1 else None
    # The official local-model CLI leaves tail fallback disabled by default.
    return None


def score_spatialviz(target: str, output: str) -> dict[str, Any]:
    prediction = extract_choice_letter(output)
    return {
        "prediction": prediction,
        "target": target,
        "score": 1.0 if prediction == target else 0.0,
        "correct": prediction == target,
        "scoring": "official_spatialviz_strict_option_accuracy",
    }


def vsi_first_word(value: str) -> str:
    parts = value.split(" ")
    return (parts[0] if parts else "").rstrip(".").strip()


def vsi_mean_relative_accuracy(
    prediction: float | None,
    target: float | None,
) -> float:
    if (
        prediction is None
        or target is None
        or not math.isfinite(prediction)
        or not math.isfinite(target)
        or target == 0
    ):
        return 0.0
    # Reproduce the official VSI-Bench implementation exactly. Its published
    # helper creates 11 thresholds with linspace from 0.50 through 0.95.
    thresholds = np.linspace(0.5, 0.95, 11)
    relative_error = abs(prediction - target) / target
    return float(np.mean(relative_error <= 1.0 - thresholds))


def score_vsi(
    question_type: str,
    target: str,
    output: str,
) -> dict[str, Any]:
    first = vsi_first_word(output)
    if question_type in VSI_MCA_TYPES:
        correct = first.lower() == target.lower()
        return {
            "prediction": first,
            "target": target,
            "score": 1.0 if correct else 0.0,
            "correct": correct,
            "scoring": "official_mca_exact_first_word",
        }
    try:
        prediction_number = float(first)
    except (TypeError, ValueError):
        prediction_number = None
    try:
        target_number = float(target)
    except (TypeError, ValueError):
        target_number = None
    score = vsi_mean_relative_accuracy(prediction_number, target_number)
    return {
        "prediction": prediction_number,
        "target": target_number,
        "score": score,
        "correct": None,
        "scoring": "official_mra_0.50_0.95",
    }


def aggregate_spatialviz(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "examples": len(rows),
        "accuracy": float(np.mean([row["score"] for row in rows])),
        "correct": int(sum(row["score"] for row in rows)),
    }


def aggregate_vsi(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(float(row["score"]))
    raw = {
        question_type: float(np.mean(scores))
        for question_type, scores in sorted(by_type.items())
    }
    direction_types = {
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
    }
    reported = {
        task: raw[task]
        for task in VSI_REPORTED_TASKS
        if task != "object_rel_direction" and task in raw
    }
    if direction_types <= set(raw):
        reported["object_rel_direction"] = float(
            np.mean([raw[task] for task in sorted(direction_types)])
        )
    missing = sorted(
        (VSI_MCA_TYPES | VSI_NUMERIC_TYPES) - set(raw)
    )
    return {
        "examples": len(rows),
        "overall_macro": (
            float(np.mean([reported[task] for task in VSI_REPORTED_TASKS]))
            if set(reported) == set(VSI_REPORTED_TASKS)
            else None
        ),
        "reported_tasks": {
            task: reported[task]
            for task in VSI_REPORTED_TASKS
            if task in reported
        },
        "raw_question_types": raw,
        "missing_question_types": missing,
    }


def inject_language_lora(
    model: torch.nn.Module,
    adapter_path: Path,
) -> dict[str, Any]:
    """Inject a text-trained adapter only into the multimodal language model."""
    config = PeftConfig.from_pretrained(adapter_path)
    if hasattr(model, "language_model"):
        owner = model
    elif hasattr(model, "model") and hasattr(model.model, "language_model"):
        owner = model.model
    else:
        raise RuntimeError("Multimodal model does not expose language_model")
    language = owner.language_model
    injection_config = config
    if (
        isinstance(config.target_modules, str)
        and config.target_modules.startswith(r"^model\.")
        and not hasattr(language, "model")
    ):
        injection_config = copy.deepcopy(config)
        injection_config.target_modules = "^" + config.target_modules.removeprefix(
            r"^model\."
        )
    language_model = inject_adapter_in_model(
        injection_config,
        language,
        adapter_name="default",
    )
    owner.language_model = language_model

    state_path = adapter_path / "adapter_model.safetensors"
    state = load_file(state_path, device="cpu")
    loaded: set[str] = set()
    modules: set[str] = set()
    for key, tensor in state.items():
        match = ADAPTER_KEY.fullmatch(key)
        if match is None:
            raise RuntimeError(f"Unexpected adapter key: {key}")
        module_path, side = match.groups()
        try:
            module = owner.language_model.get_submodule(module_path)
            normalized_module_path = module_path
        except AttributeError:
            candidates = [f"model.{module_path}"]
            if module_path.startswith("language_model."):
                candidates.append(module_path.removeprefix("language_model."))
            if module_path.startswith("model."):
                candidates.append(module_path.removeprefix("model."))
            module = None
            for candidate in candidates:
                try:
                    module = owner.language_model.get_submodule(candidate)
                    normalized_module_path = candidate
                    break
                except AttributeError:
                    continue
            if module is None:
                raise
        weights = module.lora_A if side == "A" else module.lora_B
        target = weights["default"].weight
        if target.shape != tensor.shape:
            raise RuntimeError(
                f"LoRA shape mismatch for {key}: "
                f"{tuple(target.shape)} != {tuple(tensor.shape)}"
            )
        target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
        loaded.add(key)
        modules.add(normalized_module_path)
    if loaded != set(state):
        raise RuntimeError("Not all LoRA tensors were loaded")
    # Model depth varies across the 2B/4B/9B/27B scaling family.  Every
    # targeted module contributes one A and one B tensor, so validate that
    # invariant instead of hard-coding the 9B-only 496/248 counts.
    if not modules or len(loaded) != 2 * len(modules):
        raise RuntimeError(
            "Incomplete LoRA module pairs: "
            f"{len(loaded)} tensors for {len(modules)} modules"
        )
    return {
        "adapter": str(adapter_path),
        "adapter_sha256": sha256_file(state_path),
        "lora_tensors": len(loaded),
        "lora_modules": len(modules),
        "vision_modules_modified": 0,
    }


def assign_vsi_scenes(
    rows: list[dict[str, Any]],
    world_size: int,
) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["cache_key"]].append(row)
    loads = [0 for _ in range(world_size)]
    assignment: dict[str, int] = {}
    ordered = sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    for cache_key, scene_rows in ordered:
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        assignment[cache_key] = rank
        loads[rank] += len(scene_rows)
    if max(loads) - min(loads) > 2:
        raise RuntimeError(f"VSI scene partition is unexpectedly imbalanced: {loads}")
    return assignment
