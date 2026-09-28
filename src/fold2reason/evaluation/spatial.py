#!/usr/bin/env python3
"""Full image/video evaluation for SpatialViz-Bench and VSI-Bench."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling

from fold2reason.evaluation.multimodal import (
    aggregate_spatialviz,
    aggregate_vsi,
    assign_vsi_scenes,
    expand_cached_media_tokens,
    inject_language_lora,
    load_spatialviz_rows,
    load_vsi_rows,
    multimodal_messages,
    score_spatialviz,
    score_vsi,
    sha256_file,
    spatialviz_prompt,
    vsi_prompt,
)
from fold2reason.evaluation.sampling import (
    IndependentStreamSampler,
    LOGIT_QUANTIZATION_STEP,
    canonical_prediction,
    item_summary,
    stream_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3.5-9B"),
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--spatialviz-root",
        type=Path,
        default=Path(
            "data/benchmarks/data/external/spatialviz_bench"
        ),
    )
    parser.add_argument(
        "--vsi-root",
        type=Path,
        default=Path(
            "data/benchmarks/data/external/vsi_bench"
        ),
    )
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument(
        "--selection-source",
        type=Path,
        help=(
            "Legacy general_benchs_all directory whose base-rank shards "
            "define the exact SpatialViz/VSI 1,000-question subsets."
        ),
    )
    parser.add_argument("--spatialviz-batch-size", type=int, default=2)
    parser.add_argument("--vsi-batch-size", type=int, default=2)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=("spatialviz", "vsi"),
        default=("spatialviz", "vsi"),
    )
    # The official answer-only protocol should finish well within the same
    # 128-token limit used by the preceding general-bench evaluation.
    parser.add_argument(
        "--spatialviz-max-new-tokens",
        type=int,
        default=128,
    )
    # VSI-Bench's official lmms-eval task uses 16 greedy output tokens.
    parser.add_argument("--vsi-max-new-tokens", type=int, default=16)
    parser.add_argument("--limit-spatialviz", type=int, default=0)
    parser.add_argument("--limit-vsi", type=int, default=0)
    parser.add_argument("--sample-draws", type=int, default=0)
    parser.add_argument("--draw-batch-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--eval-seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_legacy_selection(
    source: Path,
) -> tuple[set[str], set[str], dict[str, Any]]:
    shard_paths = sorted(source.glob("base-rank-*.jsonl"))
    if len(shard_paths) != 4:
        raise RuntimeError(
            f"Expected four legacy base shards in {source}, "
            f"found {len(shard_paths)}"
        )
    legacy_ids: dict[str, list[str]] = {
        "spatialviz_bench_text_only": [],
        "vsi_bench_text_only": [],
    }
    for path in shard_paths:
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            dataset = row.get("dataset")
            if dataset in legacy_ids:
                legacy_ids[dataset].append(str(row["id"]))
    for dataset, ids in legacy_ids.items():
        if len(ids) != 1000 or len(set(ids)) != 1000:
            raise RuntimeError(
                f"{dataset} legacy selection must contain 1,000 unique "
                f"IDs, got {len(ids)}/{len(set(ids))}"
            )
        legacy_ids[dataset] = sorted(ids)

    spatial_ids = {
        value.replace(
            "spatialviz_bench_text_only:", "spatialviz_multimodal:", 1
        )
        for value in legacy_ids["spatialviz_bench_text_only"]
    }
    vsi_ids = {
        value.replace("vsi_bench_text_only:", "vsi_multimodal:", 1)
        for value in legacy_ids["vsi_bench_text_only"]
    }
    canonical = "\n".join(
        f"{dataset}\t{value}"
        for dataset in sorted(legacy_ids)
        for value in legacy_ids[dataset]
    ) + "\n"
    selection_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    audit = {
        "policy": (
            "Exact ID reuse from the previous text-only evaluation; "
            "only the missing image/video modality is restored."
        ),
        "source_directory": str(source),
        "source_shards": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in shard_paths
        ],
        "legacy_id_sha256": selection_sha256,
        "legacy_ids": legacy_ids,
        "multimodal_ids": {
            "spatialviz": sorted(spatial_ids),
            "vsi": sorted(vsi_ids),
        },
    }
    return spatial_ids, vsi_ids, audit


def left_pad(
    sequences: list[list[int]],
    pad_token_id: int,
    image_token_id: int,
    video_token_id: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    width = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(sequences), width),
        dtype=torch.long,
        device=device,
    )
    for index, sequence in enumerate(sequences):
        values = torch.tensor(sequence, dtype=torch.long, device=device)
        input_ids[index, -len(sequence) :] = values
        attention_mask[index, -len(sequence) :] = 1
    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[input_ids == image_token_id] = 1
    mm_token_type_ids[input_ids == video_token_id] = 2
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
    }


def install_cached_feature_hooks(
    model: Qwen3_5ForConditionalGeneration,
) -> None:
    model.model._cached_image_features = ()
    model.model._cached_video_features = ()

    def get_images(
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        del pixel_values, image_grid_thw, kwargs
        return BaseModelOutputWithPooling(
            pooler_output=model.model._cached_image_features
        )

    def get_videos(
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        del pixel_values_videos, video_grid_thw, kwargs
        return BaseModelOutputWithPooling(
            pooler_output=model.model._cached_video_features
        )

    model.model.get_image_features = get_images
    model.model.get_video_features = get_videos


def read_cache(path: Path, expected_type: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["format_version"] != 1:
        raise RuntimeError(f"Unsupported cache format: {path}")
    if payload["media_type"] != expected_type:
        raise RuntimeError(f"Wrong media type in {path}")
    if payload["feature_tokens"] != payload["features"].shape[0]:
        raise RuntimeError(f"Corrupt visual features: {path}")
    return payload


@torch.inference_mode()
def generate_batch(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    rows: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    media_type: str,
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    media_token_id = (
        processor.image_token_id
        if media_type == "image"
        else processor.video_token_id
    )
    prompts = (
        [spatialviz_prompt(row) for row in rows]
        if media_type == "image"
        else [vsi_prompt(row) for row in rows]
    )
    paths = (
        [row["image_path"] for row in rows]
        if media_type == "image"
        else [row["video_path"] for row in rows]
    )
    sequences = [
        expand_cached_media_tokens(
            processor,
            multimodal_messages(media_type, path, prompt),
            media_token_id,
            payload["replacement_ids"],
            enable_thinking=False,
        )
        for row, payload, path, prompt in zip(
            rows, payloads, paths, prompts
        )
    ]
    encoded = left_pad(
        sequences,
        processor.tokenizer.pad_token_id,
        model.config.image_token_id,
        model.config.video_token_id,
        device,
    )
    features = tuple(
        payload["features"].to(device=device, dtype=torch.bfloat16)
        for payload in payloads
    )
    grids = torch.cat(
        [payload["grid_thw"] for payload in payloads],
        dim=0,
    ).to(device=device, dtype=torch.long)
    dummy = torch.empty((1, 1), dtype=torch.bfloat16, device=device)
    if media_type == "image":
        model.model._cached_image_features = features
        encoded["pixel_values"] = dummy
        encoded["image_grid_thw"] = grids
    else:
        model.model._cached_video_features = features
        encoded["pixel_values_videos"] = dummy
        encoded["video_grid_thw"] = grids
    width = encoded["input_ids"].shape[1]
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    return processor.batch_decode(
        generated[:, width:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


@torch.inference_mode()
def generate_sampled_batch(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    rows: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    media_type: str,
    max_new_tokens: int,
    device: torch.device,
    draws: int,
    draw_batch_size: int,
    temperature: float,
    top_p: float,
    eval_seed: int,
) -> list[list[dict[str, Any]]]:
    """Generate independent draws while reusing frozen media features."""
    media_token_id = (
        processor.image_token_id
        if media_type == "image"
        else processor.video_token_id
    )
    prompts = (
        [spatialviz_prompt(row) for row in rows]
        if media_type == "image"
        else [vsi_prompt(row) for row in rows]
    )
    paths = (
        [row["image_path"] for row in rows]
        if media_type == "image"
        else [row["video_path"] for row in rows]
    )
    sequences = [
        expand_cached_media_tokens(
            processor,
            multimodal_messages(media_type, path, prompt),
            media_token_id,
            payload["replacement_ids"],
            enable_thinking=False,
        )
        for row, payload, path, prompt in zip(rows, payloads, paths, prompts)
    ]
    encoded = left_pad(
        sequences,
        processor.tokenizer.pad_token_id,
        model.config.image_token_id,
        model.config.video_token_id,
        device,
    )
    base_features = tuple(
        payload["features"].to(device=device, dtype=torch.bfloat16)
        for payload in payloads
    )
    base_grids = torch.cat([payload["grid_thw"] for payload in payloads], dim=0).to(
        device=device, dtype=torch.long
    )
    # Qwen3.5 expands visual inputs before get_image/video_features is called.
    # The cached-feature hook ignores the values, but the leading dimension must
    # still match the sum of grid token volumes so expansion can split per sample
    # when num_return_sequences > 1.
    dummy_rows = int(base_grids.prod(dim=1).sum().item())
    dummy = torch.empty(
        (dummy_rows, 1), dtype=torch.bfloat16, device=device
    )
    width = encoded["input_ids"].shape[1]
    by_row: list[list[dict[str, Any]]] = [[] for _ in rows]
    for start in range(0, draws, draw_batch_size):
        indices = list(range(start, min(start + draw_batch_size, draws)))
        seeds = [
            stream_seed(row["id"], draw_index, eval_seed)
            for row in rows
            for draw_index in indices
        ]
        repeats = len(indices)
        expanded_features = tuple(
            feature for feature in base_features for _ in range(repeats)
        )
        call_encoded = dict(encoded)
        if media_type == "image":
            model.model._cached_image_features = expanded_features
            call_encoded["pixel_values"] = dummy
            call_encoded["image_grid_thw"] = base_grids
        else:
            model.model._cached_video_features = expanded_features
            call_encoded["pixel_values_videos"] = dummy
            call_encoded["video_grid_thw"] = base_grids
        generated = model.generate(
            **call_encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            num_return_sequences=repeats,
            logits_processor=[
                IndependentStreamSampler(
                    seeds,
                    temperature=temperature,
                    top_p=top_p,
                )
            ],
            use_cache=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        decoded = processor.batch_decode(
            generated[:, width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        cursor = 0
        for row_index in range(len(rows)):
            for draw_index in indices:
                by_row[row_index].append(
                    {
                        "draw_index": draw_index,
                        "stream_seed": stream_seed(
                            rows[row_index]["id"], draw_index, eval_seed
                        ),
                        "raw_output": decoded[cursor],
                    }
                )
                cursor += 1
    return by_row


def score_sampled_item(
    row: dict[str, Any],
    draws: list[dict[str, Any]],
    benchmark: str,
) -> dict[str, Any]:
    for draw in draws:
        if benchmark == "spatialviz":
            scored = score_spatialviz(row["target"], draw["raw_output"])
            correct = bool(scored["correct"])
        else:
            scored = score_vsi(
                row["question_type"], row["target"], draw["raw_output"]
            )
            correct = (
                bool(scored["correct"])
                if scored["correct"] is not None
                else float(scored["score"]) >= 1.0
            )
        draw.update(
            {
                "normalized_answer": canonical_prediction(scored["prediction"]),
                "correct": correct,
                "score": float(scored["score"]),
            }
        )
    draws.sort(key=lambda draw: draw["draw_index"])
    return {"draws": draws, **item_summary(draws)}


def grouped_pass_summary(rows: list[dict[str, Any]], key: str | None = None) -> dict[str, Any]:
    metric_names = [
        "pass_at_1", "pass_at_3", "pass_at_5", "pass_at_10", "pass_at_20",
        "majority_vote_at_1", "majority_vote_at_3", "majority_vote_at_5",
        "majority_vote_at_10", "majority_vote_at_20", "coverage_gap",
        "correct_sample_rate", "unique_normalized_answers",
        "answer_entropy_nats", "invalid_rate",
    ]

    def summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "examples": len(values),
            **{
                name: sum(float(row[name]) for row in values) / len(values)
                for name in metric_names
            },
        }

    payload = {"overall": summarize(rows)}
    if key is not None:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row)
        payload[f"by_{key}"] = {
            name: summarize(values) for name, values in sorted(groups.items())
        }
    return payload


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def collect_shards(
    directory: Path,
    world_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for rank in range(world_size):
        path = directory / f"rank-{rank:02d}.jsonl"
        rows.extend(
            json.loads(line)
            for line in path.read_text().splitlines()
            if line
        )
    if len(rows) != len({row["id"] for row in rows}):
        raise RuntimeError(f"Duplicate output IDs in {directory}")
    return sorted(rows, key=lambda row: row["id"])


def grouped_spatial_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {"overall": aggregate_spatialviz(rows)}
    for key in ("category", "task", "level"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        payload[f"by_{key}"] = {
            name: aggregate_spatialviz(values)
            for name, values in sorted(groups.items())
        }
    return payload


def vsi_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = aggregate_vsi(rows)
    debiased_rows = [row for row in rows if row["debiased"]]
    debiased = aggregate_vsi(debiased_rows)
    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        by_dataset[dataset] = aggregate_vsi(
            [row for row in rows if row["dataset"] == dataset]
        )
    return {
        "selected": selected,
        "selected_debiased": debiased,
        "by_dataset": by_dataset,
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
    torch.manual_seed(args.eval_seed)
    torch.cuda.manual_seed_all(args.eval_seed)

    manifest = json.loads(
        (args.visual_cache / "manifest.json").read_text()
    )
    if manifest["num_video_frames"] != 32:
        raise RuntimeError("Official VSI evaluation requires 32 cached frames")
    spatial_rows = load_spatialviz_rows(args.spatialviz_root)
    vsi_rows = load_vsi_rows(args.vsi_root)
    selection_audit = None
    if args.selection_source is not None:
        spatial_ids, vsi_ids, selection_audit = load_legacy_selection(
            args.selection_source
        )
        spatial_rows = [
            row for row in spatial_rows if row["id"] in spatial_ids
        ]
        vsi_rows = [row for row in vsi_rows if row["id"] in vsi_ids]
        if {row["id"] for row in spatial_rows} != spatial_ids:
            raise RuntimeError("Not all legacy SpatialViz IDs were resolved")
        if {row["id"] for row in vsi_rows} != vsi_ids:
            raise RuntimeError("Not all legacy VSI IDs were resolved")
        if not all(row["debiased"] for row in vsi_rows):
            raise RuntimeError(
                "Legacy VSI selection must be entirely debiased"
            )
    if args.limit_spatialviz:
        spatial_rows = spatial_rows[: args.limit_spatialviz]
    if args.limit_vsi:
        vsi_rows = vsi_rows[: args.limit_vsi]
    selection_reference = None
    if selection_audit is not None:
        selection_reference = {
            "policy": selection_audit["policy"],
            "source_directory": selection_audit["source_directory"],
            "legacy_id_sha256": selection_audit["legacy_id_sha256"],
            "spatialviz_ids": len(
                selection_audit["multimodal_ids"]["spatialviz"]
            ),
            "vsi_ids": len(selection_audit["multimodal_ids"]["vsi"]),
            "full_manifest": str(
                args.output / "selection_manifest.json"
            ),
        }

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
    )
    processor.tokenizer.padding_side = "left"
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config._attn_implementation = "sdpa"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": local_rank},
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    adapter_audit = None
    if args.adapter is not None:
        adapter_audit = inject_language_lora(model, args.adapter)
    install_cached_feature_hooks(model)

    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0 and selection_audit is not None:
        (args.output / "selection_manifest.json").write_text(
            json.dumps(selection_audit, indent=2, sort_keys=True) + "\n"
        )
    timings = {}

    if "spatialviz" in args.benchmarks:
        local_spatial = spatial_rows[rank::world_size]
        spatial_results = []
        started = time.perf_counter()
        for start in range(0, len(local_spatial), args.spatialviz_batch_size):
            batch = local_spatial[start : start + args.spatialviz_batch_size]
            payloads = [
                read_cache(args.visual_cache / row["cache_key"], "image")
                for row in batch
            ]
            if args.sample_draws:
                sampled = generate_sampled_batch(
                    model, processor, batch, payloads, "image",
                    args.spatialviz_max_new_tokens, device,
                    args.sample_draws, args.draw_batch_size,
                    args.temperature, args.top_p, args.eval_seed,
                )
                for row, draws in zip(batch, sampled):
                    spatial_results.append(
                        {
                            "id": row["id"],
                            "category": row["category"],
                            "task": row["task"],
                            "level": row["level"],
                            "image_id": row["image_id"],
                            "choice_count": 4,
                            **score_sampled_item(row, draws, "spatialviz"),
                        }
                    )
            else:
                outputs = generate_batch(
                    model, processor, batch, payloads, "image",
                    args.spatialviz_max_new_tokens, device,
                )
                for row, output in zip(batch, outputs):
                    spatial_results.append(
                        {
                            "id": row["id"],
                            "category": row["category"],
                            "task": row["task"],
                            "level": row["level"],
                            "image_id": row["image_id"],
                            "raw_output": output,
                            **score_spatialviz(row["target"], output),
                        }
                    )
            if (
                start == 0
                or (start // args.spatialviz_batch_size + 1) % 20 == 0
                or start + args.spatialviz_batch_size >= len(local_spatial)
            ):
                print(
                    f"[spatialviz:{args.model_name}] rank{rank} "
                    f"{min(start + args.spatialviz_batch_size, len(local_spatial))}"
                    f"/{len(local_spatial)}",
                    flush=True,
                )
        timings["spatialviz"] = time.perf_counter() - started
        write_rows(
            args.output / "spatialviz" / f"rank-{rank:02d}.jsonl",
            spatial_results,
        )
        dist.barrier()

    if "vsi" in args.benchmarks:
        scene_assignment = assign_vsi_scenes(vsi_rows, world_size)
        local_vsi = [
            row
            for row in vsi_rows
            if scene_assignment[row["cache_key"]] == rank
        ]
        grouped_scenes: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in local_vsi:
            grouped_scenes[row["cache_key"]].append(row)
        vsi_results = []
        started = time.perf_counter()
        completed = 0
        for scene_index, cache_key in enumerate(
            sorted(grouped_scenes), start=1
        ):
            scene_rows = sorted(
                grouped_scenes[cache_key],
                key=lambda row: row["id"],
            )
            payload = read_cache(args.visual_cache / cache_key, "video")
            for start in range(0, len(scene_rows), args.vsi_batch_size):
                batch = scene_rows[start : start + args.vsi_batch_size]
                if args.sample_draws:
                    sampled = generate_sampled_batch(
                        model, processor, batch, [payload for _ in batch],
                        "video", args.vsi_max_new_tokens, device,
                        args.sample_draws, args.draw_batch_size,
                        args.temperature, args.top_p, args.eval_seed,
                    )
                    for row, draws in zip(batch, sampled):
                        vsi_results.append(
                            {
                                "id": row["id"],
                                "numeric_id": row["numeric_id"],
                                "dataset": row["dataset"],
                                "scene_name": row["scene_name"],
                                "question_type": row["question_type"],
                                "debiased": row["debiased"],
                                **score_sampled_item(row, draws, "vsi"),
                            }
                        )
                else:
                    outputs = generate_batch(
                        model, processor, batch, [payload for _ in batch],
                        "video", args.vsi_max_new_tokens, device,
                    )
                    for row, output in zip(batch, outputs):
                        vsi_results.append(
                            {
                                "id": row["id"],
                                "numeric_id": row["numeric_id"],
                                "dataset": row["dataset"],
                                "scene_name": row["scene_name"],
                                "question_type": row["question_type"],
                                "debiased": row["debiased"],
                                "raw_output": output,
                                **score_vsi(
                                    row["question_type"], row["target"], output
                                ),
                            }
                        )
                completed += len(batch)
            if scene_index == 1 or scene_index % 2 == 0:
                print(
                    f"[vsi:{args.model_name}] rank{rank} "
                    f"{completed}/{len(local_vsi)} "
                    f"scenes={scene_index}/{len(grouped_scenes)}",
                    flush=True,
                )
        timings["vsi"] = time.perf_counter() - started
        write_rows(
            args.output / "vsi" / f"rank-{rank:02d}.jsonl",
            vsi_results,
        )
        dist.barrier()

    timing_tensor = torch.tensor(
        [
            timings.get("spatialviz", 0.0),
            timings.get("vsi", 0.0),
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(timing_tensor, op=dist.ReduceOp.MAX)
    if rank == 0:
        all_spatial = (
            collect_shards(args.output / "spatialviz", world_size)
            if "spatialviz" in args.benchmarks
            else []
        )
        all_vsi = (
            collect_shards(args.output / "vsi", world_size)
            if "vsi" in args.benchmarks
            else []
        )
        selected_spatial = 1000 if selection_audit is not None else 1180
        selected_vsi = 1000 if selection_audit is not None else 5130
        expected_spatial = (
            min(args.limit_spatialviz, selected_spatial)
            if args.limit_spatialviz
            else selected_spatial
        )
        expected_vsi = (
            min(args.limit_vsi, selected_vsi)
            if args.limit_vsi
            else selected_vsi
        )
        if (
            "spatialviz" in args.benchmarks
            and len(all_spatial) != expected_spatial
        ):
            raise RuntimeError("SpatialViz output count mismatch")
        if "vsi" in args.benchmarks and len(all_vsi) != expected_vsi:
            raise RuntimeError("VSI output count mismatch")
        spatial_summary = None
        if "spatialviz" in args.benchmarks:
            decoding = {
                "greedy": not bool(args.sample_draws),
                "enable_thinking": False,
                "max_new_tokens": args.spatialviz_max_new_tokens,
            }
            if args.sample_draws:
                decoding.update(
                    {
                        "draws": args.sample_draws,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "eval_seed": args.eval_seed,
                        "seed_derivation": "SHA256(item_id, draw_index, eval_seed)",
                        "cached_media_features_reused_across_draws": True,
                        "logit_quantization_step": LOGIT_QUANTIZATION_STEP,
                        "model_construction_seed": args.eval_seed,
                    }
                )
            spatial_summary = {
                "benchmark": "SpatialViz-Bench",
                "modality": "image",
                "official_examples": 1180,
                "evaluated_examples": len(all_spatial),
                "selection": selection_reference,
                "model_name": args.model_name,
                "prompt_protocol": "official choice_answer_only",
                "decoding": decoding,
                "answer_extraction": (
                    "official strict tagged/pattern extraction; "
                    "tail fallback disabled"
                ),
                **(
                    grouped_pass_summary(all_spatial, "task")
                    if args.sample_draws
                    else grouped_spatial_summary(all_spatial)
                ),
            }
            (args.output / "spatialviz" / "summary.json").write_text(
                json.dumps(spatial_summary, indent=2, sort_keys=True) + "\n"
            )
        vsi_results_summary = None
        if "vsi" in args.benchmarks:
            decoding = {
                "greedy": not bool(args.sample_draws),
                "enable_thinking": False,
                "max_new_tokens": args.vsi_max_new_tokens,
            }
            if args.sample_draws:
                decoding.update(
                    {
                        "draws": args.sample_draws,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "eval_seed": args.eval_seed,
                        "seed_derivation": "SHA256(item_id, draw_index, eval_seed)",
                        "cached_media_features_reused_across_draws": True,
                        "logit_quantization_step": LOGIT_QUANTIZATION_STEP,
                        "model_construction_seed": args.eval_seed,
                        "numeric_correctness": "official MRA score == 1.0 (<=5% relative error)",
                    }
                )
            vsi_results_summary = {
                "benchmark": "VSI-Bench",
                "modality": "video",
                "official_examples_full": 5130,
                "official_examples_debiased": 2362,
                "evaluated_examples": len(all_vsi),
                "selection": selection_reference,
                "model_name": args.model_name,
                "video_sampling": {
                    "uniform_frames": manifest["num_video_frames"],
                },
                "decoding": decoding,
                "official_scoring": {
                    "multiple_choice": "exact first-word accuracy",
                    "numeric": "MRA thresholds 0.50..0.95",
                    "overall": "macro mean over 8 reported tasks",
                },
                **(
                    grouped_pass_summary(all_vsi, "question_type")
                    if args.sample_draws
                    else vsi_summary(all_vsi)
                ),
            }
            (args.output / "vsi" / "summary.json").write_text(
                json.dumps(
                    vsi_results_summary,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        run_summary = {
            "model": str(args.model),
            "model_name": args.model_name,
            "adapter_audit": adapter_audit,
            "visual_cache": str(args.visual_cache),
            "visual_cache_manifest": manifest,
            "selection": selection_reference,
            "benchmarks": list(args.benchmarks),
            "spatialviz": (
                spatial_summary["overall"]
                if spatial_summary is not None
                else None
            ),
            "vsi_selected": (
                (
                    vsi_results_summary["overall"]
                    if args.sample_draws
                    else vsi_results_summary["selected"]
                )
                if vsi_results_summary is not None
                else None
            ),
            "vsi_selected_debiased": (
                (
                    None
                    if args.sample_draws
                    else vsi_results_summary["selected_debiased"]
                )
                if vsi_results_summary is not None
                else None
            ),
            "wall_seconds_max_rank": {
                "spatialviz": float(timing_tensor[0].item()),
                "vsi": float(timing_tensor[1].item()),
            },
        }
        (args.output / "run_summary.json").write_text(
            json.dumps(run_summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(run_summary, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
