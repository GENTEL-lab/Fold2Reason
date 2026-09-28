#!/usr/bin/env python3
"""Distributed GeoHead-AllAttn LoRA training for the 1k-v2 pilot."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import peft
import torch
import torch.distributed as dist
import transformers
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from fold2reason.evaluation.geometry import evaluate_rows
from fold2reason.models.geometry import (
    GeometryModel,
    add_all_attention_lora,
    load_lora,
    load_text_model,
    module_match_summary,
)
from fold2reason.losses.geometry import geometry_loss


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
    parser.add_argument(
        "--output",
        type=Path,
        default=project
        / "outputs"
        / "qwen35_9b_openfold_1k_v2"
        / "geohead"
        / "seed-20260729",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--distogram-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--fast-eval-examples", type=int, default=24)
    parser.add_argument("--final-eval-max-examples", type=int, default=0)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--coordinate-mode", choices=("free", "internal"), default="free")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--loss-aligned-coord", type=float, default=1.0)
    parser.add_argument("--loss-pair-distance", type=float, default=1.0)
    parser.add_argument("--loss-contact", type=float, default=0.5)
    parser.add_argument("--loss-distogram", type=float, default=0.3)
    parser.add_argument("--loss-backbone-local", type=float, default=0.2)
    parser.add_argument("--loss-torsion", type=float, default=0.2)
    parser.add_argument("--loss-radius-of-gyration", type=float, default=0.1)
    parser.add_argument("--loss-input-contrastive", type=float, default=0.0)
    parser.add_argument(
        "--bridge-arm",
        choices=("none", "real", "shuffled", "ordinary"),
        default="none",
        help="Optional Phase-1 native LM supervision stored in row['bridge'].",
    )
    parser.add_argument("--loss-relation", type=float, default=1.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.5)
    parser.add_argument(
        "--contrastive-mode",
        choices=("hard", "structured", "both"),
        default="both",
    )
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--freeze-language-model",
        action="store_true",
        help=(
            "Freeze the base text tower and the zero-initialized LoRA adapter, "
            "training only the geometry heads. This is the Phase-0 head-only "
            "causal control. It cannot be combined with --init-checkpoint."
        ),
    )
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 1:
        raise RuntimeError(f"Expected at least one GPU, got {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    return rank, local_rank, world_size, device


def seed_everything(seed: int, rank: int) -> None:
    actual = seed + rank
    random.seed(actual)
    np.random.seed(actual)
    torch.manual_seed(actual)
    torch.cuda.manual_seed_all(actual)


def json_dump(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def build_epoch_groups(
    rows: list[dict[str, Any]],
    world_size: int,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    ordered = sorted(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))
    usable = len(ordered) - len(ordered) % world_size
    ordered = ordered[:usable]
    groups = [
        ordered[start : start + world_size]
        for start in range(0, len(ordered), world_size)
    ]
    random.Random(seed + epoch).shuffle(groups)
    return groups


def environment_info() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_gib": round(
                    torch.cuda.get_device_properties(index).total_memory
                    / 2**30,
                    3,
                ),
            }
            for index in range(torch.cuda.device_count())
        ],
    }


def save_checkpoint(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output: Path,
    name: str,
    epoch: int,
    global_step: int,
    metrics: dict[str, Any] | None = None,
) -> Path:
    checkpoint = output / "checkpoints" / name
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.module.save_geometry_pretrained(checkpoint)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
        },
        checkpoint / "trainer_state.pt",
    )
    return checkpoint


def composite_score(metrics: dict[str, Any]) -> float:
    return (
        0.35 * metrics["tm_score"]
        + 0.35 * metrics["contact_f1"]
        + 0.20 * metrics["lddt_ca"]
        + 0.10 * metrics["ca_step_valid_fraction"]
        - 0.10 * abs(math.log(max(metrics["rg_ratio"], 1e-8)))
    )


def reduce_components(
    components: dict[str, torch.Tensor],
    world_size: int,
) -> dict[str, float]:
    names = sorted(components)
    tensor = torch.stack(
        [components[name].detach().float() for name in names]
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return {
        name: float(value)
        for name, value in zip(names, tensor.cpu().tolist())
    }


def main() -> None:
    args = parse_args()
    if args.freeze_language_model and args.init_checkpoint:
        raise ValueError(
            "--freeze-language-model cannot be combined with --init-checkpoint"
        )
    rank, local_rank, world_size, device = setup_distributed()
    seed_everything(args.seed, rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    log_path = args.output / "training_log.jsonl"

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_rows = cache["splits"]["train"]
    if args.bridge_arm != "none":
        missing_bridge = [row["id"] for row in train_rows if args.bridge_arm not in row.get("bridge", {})]
        if missing_bridge:
            raise RuntimeError(
                f"Bridge arm {args.bridge_arm!r} missing from {len(missing_bridge)} rows; "
                f"first={missing_bridge[0]}"
            )
    if args.overfit_samples:
        train_rows = sorted(
            train_rows, key=lambda row: len(row["input_ids"])
        )[: args.overfit_samples]
    if len(train_rows) < world_size:
        raise RuntimeError("Not enough rows for distributed training")
    if len(train_rows) % world_size:
        train_rows = train_rows[: len(train_rows) - len(train_rows) % world_size]

    if rank == 0:
        print(
            f"[load] model={args.model} rows={len(train_rows)} "
            f"rank={rank}/{world_size}",
            flush=True,
        )
    base, loading_info = load_text_model(args.model, device)
    if args.init_checkpoint:
        lora = load_lora(
            base,
            args.init_checkpoint / "adapter",
            trainable=True,
            gradient_checkpointing=not args.disable_gradient_checkpointing,
        )
    else:
        lora = add_all_attention_lora(
            base,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            gradient_checkpointing=not args.disable_gradient_checkpointing,
        )
    if args.freeze_language_model:
        for parameter in lora.parameters():
            parameter.requires_grad_(False)
        if hasattr(lora, "disable_input_require_grads"):
            lora.disable_input_require_grads()
    audit = module_match_summary(lora)
    hidden_size = int(base.config.hidden_size)
    model = GeometryModel(
        lora,
        hidden_size=hidden_size,
        coordinate_mode=args.coordinate_mode,
    )
    model.coordinate_head.to(device)
    model.distogram_head.to(device)
    if args.init_checkpoint and (args.init_checkpoint / "geometry_heads.pt").exists():
        heads = torch.load(
            args.init_checkpoint / "geometry_heads.pt",
            map_location="cpu",
            weights_only=False,
        )
        if "distogram_head" in heads:
            model.distogram_head.load_state_dict(heads["distogram_head"])
        if args.coordinate_mode == "free" and "coordinate_head" in heads:
            model.coordinate_head.load_state_dict(heads["coordinate_head"])

    lora_parameters = [
        parameter
        for parameter in model.language_model.parameters()
        if parameter.requires_grad
    ]
    coordinate_head_parameters = list(model.coordinate_head.parameters())
    distogram_head_parameters = list(model.distogram_head.parameters())
    head_parameters = coordinate_head_parameters + distogram_head_parameters
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if rank == 0:
        print(
            f"[model] trainable={trainable_parameters:,} "
            f"total={total_parameters:,} "
            f"fraction={trainable_parameters / total_parameters:.4%}",
            flush=True,
        )
        print(json.dumps(audit, indent=2, sort_keys=True), flush=True)

    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_parameters,
                "lr": args.lora_learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": coordinate_head_parameters,
                "lr": args.head_learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": distogram_head_parameters,
                "lr": args.distogram_learning_rate,
                "weight_decay": args.weight_decay,
            },
        ],
        fused=True,
    )
    groups_per_epoch = len(train_rows) // world_size
    steps_per_epoch = math.ceil(
        groups_per_epoch / args.gradient_accumulation
    )
    scheduled_steps = steps_per_epoch * args.epochs
    if args.max_optimizer_steps:
        scheduled_steps = min(
            scheduled_steps, args.max_optimizer_steps
        )
    warmup_steps = max(1, round(scheduled_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=scheduled_steps,
    )

    run_config = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": environment_info(),
        "world_size": world_size,
        "cache_stats": cache["stats"],
        "model_loading": loading_info,
        "lora_audit": audit,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "coordinate_mode": args.coordinate_mode,
        "freeze_language_model": args.freeze_language_model,
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "loss_weights": {
            "aligned_coord": args.loss_aligned_coord,
            "pair_distance": args.loss_pair_distance,
            "contact": args.loss_contact,
            "distogram": args.loss_distogram,
            "backbone_local": args.loss_backbone_local,
            "torsion": args.loss_torsion,
            "radius_of_gyration": args.loss_radius_of_gyration,
            "input_contrastive": args.loss_input_contrastive,
            "relation": args.loss_relation if args.bridge_arm != "none" else 0.0,
        },
        "lora_trainable_parameters": sum(
            parameter.numel() for parameter in lora_parameters
        ),
        "coordinate_head_trainable_parameters": sum(
            parameter.numel() for parameter in coordinate_head_parameters
        ),
        "distogram_head_trainable_parameters": sum(
            parameter.numel() for parameter in distogram_head_parameters
        ),
        "groups_per_epoch": groups_per_epoch,
        "steps_per_epoch": steps_per_epoch,
        "scheduled_steps": scheduled_steps,
        "warmup_steps": warmup_steps,
    }
    if rank == 0:
        json_dump(args.output / "run_config.json", run_config)
        if log_path.exists():
            log_path.unlink()
    dist.barrier()

    optimizer.zero_grad(set_to_none=True)
    loss_weights = {
        "aligned_coord": args.loss_aligned_coord,
        "pair_distance": args.loss_pair_distance,
        "contact": args.loss_contact,
        "distogram": args.loss_distogram,
        "backbone_local": args.loss_backbone_local,
        "torsion": args.loss_torsion,
        "radius_of_gyration": args.loss_radius_of_gyration,
        "input_contrastive": args.loss_input_contrastive,
    }
    model.train()
    global_step = 0
    training_started = time.perf_counter()
    best_score = -float("inf")
    best_checkpoint = None
    epochs_without_improvement = 0
    stop = False
    accumulated: dict[str, float] = {}
    accumulated_microsteps = 0
    cumulative_tokens = 0

    for epoch in range(args.epochs):
        groups = build_epoch_groups(
            train_rows, world_size, args.seed, epoch
        )
        pending = 0
        for group_index, group in enumerate(groups):
            row = train_rows[group[rank]]
            pending += 1
            is_last = group_index == len(groups) - 1
            should_sync = (
                pending == args.gradient_accumulation or is_last
            )
            sync_context = (
                contextlib.nullcontext()
                if should_sync
                else model.no_sync()
            )
            micro_started = time.perf_counter()
            input_ids = row["input_ids"].to(
                device=device, dtype=torch.long
            )
            marker_positions = row["marker_positions"].to(
                device=device, dtype=torch.long
            )
            relation_input_ids = None
            relation_labels = None
            if args.bridge_arm != "none":
                bridge = row["bridge"][args.bridge_arm]
                relation_input_ids = bridge["input_ids"].to(device=device, dtype=torch.long)
                relation_labels = bridge["labels"].to(device=device, dtype=torch.long)
            with sync_context:
                outputs = model(
                    input_ids,
                    marker_positions,
                    relation_input_ids=relation_input_ids,
                    relation_labels=relation_labels,
                )
                loss, components = geometry_loss(
                    outputs,
                    row,
                    device,
                    loss_weights=loss_weights,
                    contrastive_mode=args.contrastive_mode,
                    contrastive_margin=args.contrastive_margin,
                )
                if args.bridge_arm != "none":
                    relation_loss = outputs["relation_loss"]
                    geometry_total = components["total"]
                    loss = loss + args.loss_relation * relation_loss
                    components = {
                        **components,
                        "geometry_total": geometry_total,
                        "relation_ce": relation_loss,
                        "total": loss,
                    }
                divisor = (
                    pending
                    if is_last
                    else args.gradient_accumulation
                )
                (loss / divisor).backward()
            reduced = reduce_components(components, world_size)
            for name, value in reduced.items():
                accumulated[name] = accumulated.get(name, 0.0) + value
            accumulated_microsteps += 1
            tokens = torch.tensor(
                len(row["input_ids"])
                + (len(relation_input_ids) if relation_input_ids is not None else 0),
                dtype=torch.long,
                device=device,
            )
            dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
            cumulative_tokens += int(tokens)
            if not should_sync:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                lora_parameters + head_parameters,
                args.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            global_step += 1
            elapsed = time.perf_counter() - training_started
            averaged = {
                name: value / accumulated_microsteps
                for name, value in accumulated.items()
            }
            record = {
                "event": "train_step",
                "epoch": epoch,
                "group": group_index,
                "global_step": global_step,
                **averaged,
                "lora_learning_rate": scheduler.get_last_lr()[0],
                "head_learning_rate": scheduler.get_last_lr()[1],
                "distogram_learning_rate": scheduler.get_last_lr()[2],
                "grad_norm": float(grad_norm),
                "cumulative_input_tokens": cumulative_tokens,
                "aggregate_input_tokens_per_second": (
                    cumulative_tokens / elapsed
                ),
                "last_microstep_seconds": (
                    time.perf_counter() - micro_started
                ),
                "max_memory_allocated_gib": round(
                    torch.cuda.max_memory_allocated(device) / 2**30, 3
                ),
                "wall_seconds": elapsed,
            }
            if rank == 0:
                append_jsonl(log_path, record)
                print(
                    f"[train] step={global_step}/{scheduled_steps} "
                    f"epoch={epoch + 1} total={record['total']:.4f} "
                    f"pair={record['pair_distance']:.4f} "
                    f"contact={record['contact']:.4f} "
                    f"tm_rate={record['aggregate_input_tokens_per_second']:.0f} "
                    f"tok/s mem={record['max_memory_allocated_gib']:.1f}GiB",
                    flush=True,
                )
            accumulated = {}
            accumulated_microsteps = 0

            checkpoint_due = (
                args.checkpoint_every
                and global_step % args.checkpoint_every == 0
            )
            if checkpoint_due:
                dist.barrier()
                if rank == 0:
                    checkpoint = save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        args.output,
                        f"step-{global_step:04d}",
                        epoch,
                        global_step,
                    )
                    print(f"[checkpoint] {checkpoint}", flush=True)
                dist.barrier()

            if (
                args.fast_eval_examples
                and global_step % 50 == 0
                and global_step < scheduled_steps
            ):
                metrics = evaluate_rows(
                    model,
                    cache["splits"]["validation"],
                    f"validation-fast-step-{global_step}",
                    rank,
                    world_size,
                    device,
                    max_examples=args.fast_eval_examples,
                )
                if rank == 0:
                    append_jsonl(
                        log_path,
                        {
                            "event": "fast_evaluation",
                            "global_step": global_step,
                            **metrics,
                        },
                    )
                    print(
                        f"[fast-eval] step={global_step} "
                        f"TM={metrics['tm_score']:.4f} "
                        f"contactF1={metrics['contact_f1']:.4f}",
                        flush=True,
                    )
                model.train()

            if (
                args.max_optimizer_steps
                and global_step >= args.max_optimizer_steps
            ):
                stop = True
                break

        epoch_evaluation_rows = (
            train_rows if args.overfit_samples else cache["splits"]["validation"]
        )
        epoch_evaluation_limit = (
            min(len(train_rows), 24) if args.overfit_samples else 0
        )
        epoch_metrics = evaluate_rows(
            model,
            epoch_evaluation_rows,
            (
                f"overfit-train-epoch-{epoch + 1}"
                if args.overfit_samples
                else f"validation-epoch-{epoch + 1}"
            ),
            rank,
            world_size,
            device,
            max_examples=epoch_evaluation_limit,
        )
        score = composite_score(epoch_metrics)
        if rank == 0:
            append_jsonl(
                log_path,
                {
                    "event": "epoch_evaluation",
                    "epoch": epoch,
                    "global_step": global_step,
                    "composite_score": score,
                    **epoch_metrics,
                },
            )
            checkpoint = save_checkpoint(
                model,
                optimizer,
                scheduler,
                args.output,
                f"epoch-{epoch + 1:02d}",
                epoch,
                global_step,
                metrics=epoch_metrics,
            )
            if score > best_score:
                best_score = score
                best_checkpoint = save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    args.output,
                    "best",
                    epoch,
                    global_step,
                    metrics=epoch_metrics,
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            print(
                f"[epoch-eval] epoch={epoch + 1} "
                f"TM={epoch_metrics['tm_score']:.4f} "
                f"contactF1={epoch_metrics['contact_f1']:.4f} "
                f"score={score:.4f} checkpoint={checkpoint}",
                flush=True,
            )
        score_tensor = torch.tensor(score, device=device)
        dist.broadcast(score_tensor, src=0)
        patience_tensor = torch.tensor(
            epochs_without_improvement,
            dtype=torch.long,
            device=device,
        )
        dist.broadcast(patience_tensor, src=0)
        epochs_without_improvement = int(patience_tensor.item())
        model.train()
        if stop or epochs_without_improvement >= 2:
            break

    dist.barrier()
    if rank == 0:
        final_checkpoint = args.output / "final"
        model.module.save_geometry_pretrained(final_checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True
        )
        tokenizer.save_pretrained(final_checkpoint / "adapter")
        print(f"[save] final={final_checkpoint}", flush=True)
    dist.barrier()

    final_evaluations = {}
    for split in (
        name
        for name in ("validation", "validation_rare")
        if name in cache["splits"]
    ):
        final_evaluations[split] = evaluate_rows(
            model,
            cache["splits"][split],
            split,
            rank,
            world_size,
            device,
            max_examples=args.final_eval_max_examples,
        )
    wall_seconds = time.perf_counter() - training_started
    if rank == 0:
        summary = {
            "status": (
                "SMOKE_COMPLETE"
                if args.max_optimizer_steps
                and global_step < steps_per_epoch * args.epochs
                else "TRAINING_COMPLETE"
            ),
            "global_steps": global_step,
            "training_wall_seconds": wall_seconds,
            "best_composite_score": best_score,
            "best_checkpoint": (
                str(best_checkpoint) if best_checkpoint else None
            ),
            "final_checkpoint": str(args.output / "final"),
            "evaluations": final_evaluations,
            "max_memory_allocated_gib_rank0": round(
                torch.cuda.max_memory_allocated(device) / 2**30, 3
            ),
        }
        json_dump(args.output / "run_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
