#!/usr/bin/env python3
"""Build reusable Qwen3.5 vision features for full SpatialViz and VSI-Bench."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
)

from fold2reason.evaluation.multimodal import (
    SPATIALVIZ_REVISION,
    VSI_REVISION,
    load_spatialviz_rows,
    load_vsi_rows,
    multimodal_messages,
    sha256_file,
)


PROBE = "CACHE_PROBE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen3.5-9B"),
    )
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
    parser.add_argument("--num-video-frames", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def replacement_ids(
    processor: Any,
    messages: list[dict[str, Any]],
    processed_ids: list[int],
    media_token_id: int,
) -> list[int]:
    rendered = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
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
        raise RuntimeError(f"Expected one media token, found {len(positions)}")
    index = positions[0]
    prefix = raw_ids[:index]
    suffix = raw_ids[index + 1 :]
    if processed_ids[:index] != prefix:
        raise RuntimeError("Expanded multimodal prefix mismatch")
    if suffix and processed_ids[-len(suffix) :] != suffix:
        raise RuntimeError("Expanded multimodal suffix mismatch")
    end = len(processed_ids) - len(suffix) if suffix else len(processed_ids)
    return processed_ids[index:end]


@torch.inference_mode()
def build_one(
    model: Qwen3_5ForConditionalGeneration,
    processor: Any,
    media_type: str,
    media_path: str,
    cache_path: Path,
    source_id: str,
    num_video_frames: int,
) -> dict[str, Any]:
    messages = multimodal_messages(media_type, media_path, PROBE)
    kwargs: dict[str, Any] = {}
    if media_type == "video":
        kwargs["processor_kwargs"] = {
            "videos_kwargs": {
                "num_frames": num_video_frames,
                "fps": None,
            }
        }
    processed = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
        **kwargs,
    )
    device = next(model.parameters()).device
    if media_type == "image":
        pixels = processed["pixel_values"].to(device)
        grid = processed["image_grid_thw"].to(device)
        output = model.get_image_features(
            pixels,
            grid,
            return_dict=True,
        )
        media_token_id = processor.image_token_id
        token_id = model.config.image_token_id
    else:
        pixels = processed["pixel_values_videos"].to(device)
        grid = processed["video_grid_thw"].to(device)
        output = model.get_video_features(
            pixels,
            grid,
            return_dict=True,
        )
        media_token_id = processor.video_token_id
        token_id = model.config.video_token_id
    if len(output.pooler_output) != 1:
        raise RuntimeError("Expected exactly one visual feature sequence")
    features = output.pooler_output[0].detach().to("cpu", torch.bfloat16)
    expanded_ids = processed["input_ids"][0].tolist()
    replacements = replacement_ids(
        processor,
        messages,
        expanded_ids,
        media_token_id,
    )
    feature_tokens = sum(value == token_id for value in replacements)
    if feature_tokens != features.shape[0]:
        raise RuntimeError(
            f"Feature/token mismatch: {features.shape[0]} != {feature_tokens}"
        )
    payload = {
        "format_version": 1,
        "source_id": source_id,
        "media_type": media_type,
        "media_path": media_path,
        "num_video_frames": (
            num_video_frames if media_type == "video" else None
        ),
        "grid_thw": grid.detach().cpu(),
        "replacement_ids": replacements,
        "features": features,
        "feature_tokens": int(features.shape[0]),
        "hidden_size": int(features.shape[1]),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(
        f".rank-{int(os.environ['RANK']):02d}.tmp"
    )
    torch.save(payload, temporary)
    os.replace(temporary, cache_path)
    return {
        key: value
        for key, value in payload.items()
        if key not in {"features", "replacement_ids", "grid_thw"}
    } | {
        "cache_path": str(cache_path),
        "grid_thw": payload["grid_thw"].tolist(),
        "replacement_token_count": len(replacements),
        "cache_bytes": cache_path.stat().st_size,
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

    spatial_rows = load_spatialviz_rows(args.spatialviz_root)
    vsi_rows = load_vsi_rows(args.vsi_root)
    scenes: dict[str, dict[str, Any]] = {}
    for row in vsi_rows:
        scenes.setdefault(
            row["cache_key"],
            {
                "cache_key": row["cache_key"],
                "source_id": f"{row['dataset']}:{row['scene_name']}",
                "video_path": row["video_path"],
            },
        )
    image_items = [
        {
            "cache_key": row["cache_key"],
            "source_id": row["id"],
            "media_path": row["image_path"],
        }
        for row in spatial_rows
    ]
    video_items = [scenes[key] for key in sorted(scenes)]

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
    )
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

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    for media_type, items in (("image", image_items), ("video", video_items)):
        local_items = items[rank::world_size]
        for index, item in enumerate(local_items, start=1):
            cache_path = args.output / item["cache_key"]
            if cache_path.is_file() and args.resume:
                existing = torch.load(
                    cache_path,
                    map_location="cpu",
                    weights_only=False,
                )
                record = {
                    key: value
                    for key, value in existing.items()
                    if key not in {"features", "replacement_ids", "grid_thw"}
                } | {
                    "cache_path": str(cache_path),
                    "grid_thw": existing["grid_thw"].tolist(),
                    "replacement_token_count": len(
                        existing["replacement_ids"]
                    ),
                    "cache_bytes": cache_path.stat().st_size,
                    "resumed": True,
                }
            else:
                record = build_one(
                    model=model,
                    processor=processor,
                    media_type=media_type,
                    media_path=(
                        item["media_path"]
                        if media_type == "image"
                        else item["video_path"]
                    ),
                    cache_path=cache_path,
                    source_id=item["source_id"],
                    num_video_frames=args.num_video_frames,
                )
                record["resumed"] = False
            records.append(record)
            interval = 20 if media_type == "image" else 2
            if index == 1 or index % interval == 0 or index == len(local_items):
                print(
                    f"[cache:{media_type}] rank{rank} "
                    f"{index}/{len(local_items)}",
                    flush=True,
                )
    rank_manifest = args.output / f"cache-build-rank-{rank:02d}.jsonl"
    with rank_manifest.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    dist.barrier()

    elapsed = torch.tensor(
        [time.perf_counter() - started],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    if rank == 0:
        all_records = []
        for shard in range(world_size):
            path = args.output / f"cache-build-rank-{shard:02d}.jsonl"
            all_records.extend(
                json.loads(line)
                for line in path.read_text().splitlines()
                if line
            )
        image_records = [
            row for row in all_records if row["media_type"] == "image"
        ]
        video_records = [
            row for row in all_records if row["media_type"] == "video"
        ]
        if len(image_records) != 1180 or len(video_records) != 288:
            raise RuntimeError(
                f"Cache count mismatch: {len(image_records)}/"
                f"{len(video_records)}"
            )
        manifest = {
            "format_version": 1,
            "model": str(args.model),
            "model_config_sha256": sha256_file(
                args.model / "config.json"
            ),
            "preprocessor_config_sha256": sha256_file(
                args.model / "preprocessor_config.json"
            ),
            "video_preprocessor_config_sha256": sha256_file(
                args.model / "video_preprocessor_config.json"
            ),
            "spatialviz_revision": SPATIALVIZ_REVISION,
            "vsi_revision": VSI_REVISION,
            "num_video_frames": args.num_video_frames,
            "world_size": world_size,
            "wall_seconds_max_rank": float(elapsed.item()),
            "spatialviz": {
                "examples": len(spatial_rows),
                "cached_images": len(image_records),
                "feature_tokens": int(
                    sum(row["feature_tokens"] for row in image_records)
                ),
                "cache_bytes": int(
                    sum(row["cache_bytes"] for row in image_records)
                ),
            },
            "vsi": {
                "examples_full": len(vsi_rows),
                "examples_debiased": sum(
                    row["debiased"] for row in vsi_rows
                ),
                "cached_videos": len(video_records),
                "feature_tokens": int(
                    sum(row["feature_tokens"] for row in video_records)
                ),
                "cache_bytes": int(
                    sum(row["cache_bytes"] for row in video_records)
                ),
            },
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
