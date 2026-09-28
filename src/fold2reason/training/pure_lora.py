#!/usr/bin/env python3
"""Distributed Pure-LoRA SFT on frozen FoldingCorpus CE tensors."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from fold2reason.models.geometry import add_all_attention_lora, load_text_model, module_match_summary
from fold2reason.training.geometry import (
    append_jsonl,
    build_epoch_groups,
    environment_info,
    json_dump,
    reduce_components,
    seed_everything,
    setup_distributed,
)


class PureLoraRelationCEModel(nn.Module):
    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.language_model = language_model
        self.relation_forward_calls = 0

    def selected_ce_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        target_positions = torch.nonzero(labels[1:] != -100, as_tuple=False).flatten() + 1
        if len(target_positions) == 0:
            raise RuntimeError("FoldingCorpus CE row has no supervised answer positions")
        outputs = self.language_model(input_ids=input_ids.unsqueeze(0), use_cache=False)
        logits = outputs.logits[0, target_positions - 1]
        return F.cross_entropy(logits.float(), labels[target_positions])

    def full_ce_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            use_cache=False,
        ).loss

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self.relation_forward_calls += 1
        return self.selected_ce_loss(input_ids, labels)


def parse_args() -> argparse.Namespace:
    project = Path.cwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--arm", default="pure_lora_relation_ce")
    parser.add_argument(
        "--relation-variant",
        choices=("real", "shuffled"),
        default="real",
        help="Frozen Relation target used for CE; primary experiments use real.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=project / "artifacts/cache/openfold_phase2_workspace_v0.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--checkpoint-steps",
        default="",
        help="Comma-separated exact optimizer steps to save in addition to --checkpoint-every.",
    )
    parser.add_argument("--save-step-zero", action="store_true")
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=1,
        help="Save every N epoch endpoints; 0 saves only the terminal epoch endpoint.",
    )
    parser.add_argument("--stop-after-optimizer-steps", type=int, default=0)
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--skip-reference-ce-check", action="store_true")
    parser.add_argument(
        "--generic-auto-model",
        action="store_true",
        help="Load with AutoModelForCausalLM and generic decoder LoRA targets.",
    )
    return parser.parse_args()


def bridge_variant(
    row: dict[str, Any], variant: str
) -> tuple[torch.Tensor, torch.Tensor]:
    bridge = row["bridge"][variant]
    return bridge["input_ids"], bridge["labels"]


def load_output_tokenizer(model_path: Path) -> Any:
    """Load the tokenizer saved alongside the final PEFT adapter."""
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    return AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        fix_mistral_regex=getattr(config, "model_type", None) == "mistral3",
    )


def group_hash(rows: list[dict[str, Any]], groups: list[list[int]]) -> str:
    payload = [
        [str(rows[index]["id"]) for index in group]
        for group in groups
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def trainable_parameter_sha256(named_parameters: list[tuple[str, nn.Parameter]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        digest.update(name.encode())
        value = parameter.detach().cpu().float().contiguous()
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


GENERIC_DECODER_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_generic_text_model(
    model_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config._attn_implementation = "sdpa"
    config.use_cache = False
    if hasattr(config, "text_config"):
        config.text_config._attn_implementation = "sdpa"
        config.text_config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": device.index},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model, {"loader": "AutoModelForCausalLM"}


def add_generic_decoder_lora(
    base: nn.Module,
    rank: int,
    alpha: int,
    dropout: float,
    gradient_checkpointing: bool,
) -> nn.Module:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=GENERIC_DECODER_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base, config)
    model.enable_input_require_grads()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def generic_lora_module_summary(model: nn.Module) -> dict[str, Any]:
    counts = {name: 0 for name in GENERIC_DECODER_TARGET_MODULES}
    parameters = {name: 0 for name in GENERIC_DECODER_TARGET_MODULES}
    unmatched = []
    for module_name, module in model.named_modules():
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        suffix = module_name.rsplit(".", 1)[-1]
        if suffix not in counts:
            unmatched.append(module_name)
            continue
        counts[suffix] += 1
        parameters[suffix] += sum(
            parameter.numel()
            for parameter in (
                module.lora_A["default"].weight,
                module.lora_B["default"].weight,
            )
        )
    missing = [name for name, count in counts.items() if count == 0]
    if missing or unmatched:
        raise RuntimeError(
            f"Generic LoRA module audit failed: missing={missing}, "
            f"unmatched={unmatched[:8]}"
        )
    return {
        "module_counts": counts,
        "trainable_parameters_by_suffix": parameters,
        "lora_parameters": int(sum(parameters.values())),
        "target_modules": GENERIC_DECODER_TARGET_MODULES,
    }


def save_checkpoint(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output: Path,
    name: str,
    epoch: int,
    global_step: int,
    cumulative_input_tokens: int,
    elapsed_wall_seconds: float,
) -> Path:
    checkpoint = output / "checkpoints" / name
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.module.language_model.save_pretrained(checkpoint / "adapter")
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cumulative_input_tokens": cumulative_input_tokens,
            "elapsed_wall_seconds": elapsed_wall_seconds,
        },
        checkpoint / "trainer_state.pt",
    )
    return checkpoint


def main() -> None:
    args = parse_args()
    checkpoint_steps = {
        int(value)
        for value in args.checkpoint_steps.split(",")
        if value.strip()
    }
    if any(step < 0 for step in checkpoint_steps):
        raise ValueError("--checkpoint-steps must contain non-negative integers")
    if 0 in checkpoint_steps and not args.save_step_zero:
        raise ValueError("checkpoint step 0 requires --save-step-zero")
    rank, local_rank, world_size, device = setup_distributed()
    if world_size != 4:
        raise RuntimeError(f"Pure-LoRA contract requires world_size=4, got {world_size}")
    seed_everything(args.seed, rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    log_path = args.output / "training_log.jsonl"

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_rows = list(cache["splits"]["train"])
    if len(train_rows) != 1000:
        raise RuntimeError(f"Expected 1000 train rows, got {len(train_rows)}")
    if len(train_rows) % world_size:
        train_rows = train_rows[: len(train_rows) - len(train_rows) % world_size]
    relation_lengths = []
    supervised_lengths = []
    for row in train_rows:
        input_ids, labels = bridge_variant(row, args.relation_variant)
        relation_lengths.append(int(len(input_ids)))
        supervised_lengths.append(int((labels != -100).sum().item()))

    if rank == 0:
        print(
            f"[pure-lora] load model={args.model} rows={len(train_rows)} "
            f"world_size={world_size}",
            flush=True,
        )
    if args.generic_auto_model:
        base, loading_info = load_generic_text_model(args.model, device)
        lora = add_generic_decoder_lora(
            base,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            gradient_checkpointing=not args.disable_gradient_checkpointing,
        )
        lora_audit = generic_lora_module_summary(lora)
    else:
        base, loading_info = load_text_model(args.model, device)
        lora = add_all_attention_lora(
            base,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            gradient_checkpointing=not args.disable_gradient_checkpointing,
        )
        lora_audit = module_match_summary(lora)
    model = PureLoraRelationCEModel(lora).to(device)
    trainable_named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    non_lora_trainable = [
        name for name, _ in trainable_named if "lora_" not in name
    ]
    if non_lora_trainable:
        raise RuntimeError(f"Non-LoRA trainable parameters: {non_lora_trainable[:8]}")
    lora_parameters = [parameter for _, parameter in trainable_named]
    initial_trainable_sha256 = trainable_parameter_sha256(trainable_named)

    if not args.skip_reference_ce_check:
        model.eval()
        input_ids, labels = bridge_variant(train_rows[0], args.relation_variant)
        input_ids = input_ids.to(device=device, dtype=torch.long)
        labels = labels.to(device=device, dtype=torch.long)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            selected = model.selected_ce_loss(input_ids, labels)
            full = model.full_ce_loss(input_ids, labels)
        torch.testing.assert_close(selected.float(), full.float(), rtol=2e-3, atol=2e-3)
        model.train()

    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.AdamW(
        [{"params": lora_parameters, "lr": args.lora_learning_rate, "weight_decay": args.weight_decay}],
        fused=True,
    )
    groups_per_epoch = len(train_rows) // world_size
    steps_per_epoch = math.ceil(groups_per_epoch / args.gradient_accumulation)
    scheduled_steps = steps_per_epoch * args.epochs
    if args.stop_after_optimizer_steps:
        scheduled_steps = min(scheduled_steps, args.stop_after_optimizer_steps)
    warmup_steps = max(1, round(scheduled_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, scheduled_steps)

    epoch_group_hashes = {}
    rank_exposure = {
        local_rank: {
            "examples": 0,
            "relation_supervised_tokens": 0,
            "causal_lm_input_tokens": 0,
            "row_ids": [],
        }
    }
    if rank == 0:
        if log_path.exists():
            log_path.unlink()
        for epoch in range(args.epochs):
            groups = build_epoch_groups(train_rows, world_size, args.seed, epoch)
            epoch_group_hashes[f"epoch-{epoch + 1:02d}"] = group_hash(train_rows, groups)
        json_dump(
            args.output / "run_config.json",
            {
                "arm": args.arm,
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "environment": environment_info(),
                "world_size": world_size,
                "model_loading": loading_info,
                "cache_stats": cache.get("stats", {}),
                "lora_audit": lora_audit,
                "lora_trainable_parameters": sum(parameter.numel() for parameter in lora_parameters),
                "non_lora_trainable_parameters": 0,
                "initial_trainable_sha256": initial_trainable_sha256,
                "relation_input_tokens": {
                    "min": min(relation_lengths),
                    "max": max(relation_lengths),
                    "mean": sum(relation_lengths) / len(relation_lengths),
                    "per_epoch_total": sum(relation_lengths),
                },
                "relation_supervised_tokens": {
                    "min": min(supervised_lengths),
                    "max": max(supervised_lengths),
                    "mean": sum(supervised_lengths) / len(supervised_lengths),
                    "per_epoch_total": sum(supervised_lengths),
                },
                "steps_per_epoch": steps_per_epoch,
                "scheduled_steps": scheduled_steps,
                "warmup_steps": warmup_steps,
                "epoch_group_hashes": epoch_group_hashes,
                "architecture_audit": {
                    "workspace_instantiated": False,
                    "memory_token_count": 0,
                    "geometry_forward_calls": 0,
                    "retrieval_forward_calls": 0,
                    "geometry_head_parameters": 0,
                    "workspace_trainable_parameters": 0,
                    "retrieval_trainable_parameters": 0,
                },
            },
        )
    dist.barrier()

    if args.save_step_zero:
        if rank == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                args.output,
                "step-0000",
                -1,
                0,
                0,
                0.0,
            )
        dist.barrier()

    optimizer.zero_grad(set_to_none=True)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    global_step = 0
    cumulative_input_tokens = 0
    pending = 0
    accumulated: dict[str, float] = {}
    accumulated_microsteps = 0
    stop = False

    for epoch in range(args.epochs):
        groups = build_epoch_groups(train_rows, world_size, args.seed, epoch)
        for group_index, group in enumerate(groups):
            row = train_rows[group[rank]]
            input_ids_cpu, labels_cpu = bridge_variant(row, args.relation_variant)
            input_ids = input_ids_cpu.to(device=device, dtype=torch.long)
            labels = labels_cpu.to(device=device, dtype=torch.long)
            supervised = int((labels_cpu != -100).sum().item())
            rank_exposure[local_rank]["examples"] += 1
            rank_exposure[local_rank]["relation_supervised_tokens"] += supervised
            rank_exposure[local_rank]["causal_lm_input_tokens"] += int(len(input_ids_cpu))
            rank_exposure[local_rank]["row_ids"].append(str(row["id"]))
            pending += 1
            is_last = group_index == len(groups) - 1
            should_sync = pending == args.gradient_accumulation or is_last
            sync_context = contextlib.nullcontext() if should_sync else model.no_sync()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(input_ids, labels)
                (loss / (pending if is_last else args.gradient_accumulation)).backward()
            components = {
                "relation_ce": loss.detach(),
                "total": loss.detach(),
                "answer_tokens": torch.tensor(float(supervised), device=device),
            }
            reduced = reduce_components(components, world_size)
            for name, value in reduced.items():
                accumulated[name] = accumulated.get(name, 0.0) + value
            accumulated_microsteps += 1
            tokens = torch.tensor(int(len(input_ids_cpu)), device=device, dtype=torch.long)
            dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
            cumulative_input_tokens += int(tokens.item())
            if not should_sync:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(lora_parameters, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            global_step += 1
            elapsed = time.perf_counter() - started
            record = {
                "event": "train_step",
                "epoch": epoch,
                "global_step": global_step,
                **{name: value / accumulated_microsteps for name, value in accumulated.items()},
                "lora_learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "cumulative_input_tokens": cumulative_input_tokens,
                "aggregate_input_tokens_per_second": cumulative_input_tokens / max(elapsed, 1e-8),
                "max_memory_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                "wall_seconds": elapsed,
            }
            if rank == 0:
                append_jsonl(log_path, record)
                print(
                    f"[pure-lora] seed={args.seed} step={global_step}/{scheduled_steps} "
                    f"loss={record['total']:.4f} mem={record['max_memory_allocated_gib']:.1f}GiB",
                    flush=True,
                )
            accumulated = {}
            accumulated_microsteps = 0
            should_save_step = (
                (args.checkpoint_every and global_step % args.checkpoint_every == 0)
                or global_step in checkpoint_steps
            )
            if should_save_step:
                dist.barrier()
                if rank == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        args.output,
                        f"step-{global_step:04d}",
                        epoch,
                        global_step,
                        cumulative_input_tokens,
                        elapsed,
                    )
                dist.barrier()
            if args.stop_after_optimizer_steps and global_step >= args.stop_after_optimizer_steps:
                stop = True
                break
        dist.barrier()
        save_epoch_checkpoint = (
            (args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0)
            or stop
            or epoch + 1 == args.epochs
        )
        if rank == 0 and save_epoch_checkpoint:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                args.output,
                f"epoch-{epoch + 1:02d}",
                epoch,
                global_step,
                cumulative_input_tokens,
                time.perf_counter() - started,
            )
        dist.barrier()
        if stop:
            break

    if rank == 0:
        model.module.language_model.save_pretrained(args.output / "final" / "adapter")
        tokenizer = load_output_tokenizer(args.model)
        tokenizer.save_pretrained(args.output / "final" / "adapter")
    gathered_exposure: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_exposure, rank_exposure[local_rank])
    dist.barrier()
    if rank == 0:
        missing_checkpoint_steps = sorted(
            step
            for step in checkpoint_steps
            if not (args.output / "checkpoints" / f"step-{step:04d}" / "trainer_state.pt").is_file()
        )
        if missing_checkpoint_steps:
            raise RuntimeError(
                f"requested checkpoint steps were not saved: {missing_checkpoint_steps}"
            )
        final_named = [
            (name, parameter)
            for name, parameter in model.module.named_parameters()
            if parameter.requires_grad
        ]
        summary = {
            "arm": args.arm,
            "relation_variant": args.relation_variant,
            "status": "SMOKE_COMPLETE" if args.stop_after_optimizer_steps else "TRAINING_COMPLETE",
            "seed": args.seed,
            "global_steps": global_step,
            "epochs": args.epochs,
            "world_size": world_size,
            "gradient_accumulation": args.gradient_accumulation,
            "cumulative_input_tokens": cumulative_input_tokens,
            "training_wall_seconds": time.perf_counter() - started,
            "max_memory_allocated_gib_rank0": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
            "final_checkpoint": str(args.output / "final"),
            "model_loading": loading_info,
            "lora_trainable_parameters": sum(parameter.numel() for parameter in lora_parameters),
            "non_lora_trainable_parameters": 0,
            "initial_trainable_sha256": initial_trainable_sha256,
            "final_trainable_sha256": trainable_parameter_sha256(final_named),
            "workspace_instantiated": False,
            "memory_token_count": 0,
            "geometry_forward_calls": 0,
            "retrieval_forward_calls": 0,
            "relation_forward_calls_rank0": model.module.relation_forward_calls,
            "relation_ce_examples_rank0": gathered_exposure[0]["examples"],
            "relation_supervised_tokens_rank0": gathered_exposure[0]["relation_supervised_tokens"],
            "rank_exposure": {
                str(index): {
                    **exposure,
                    "row_id_sha256": hashlib.sha256(
                        "\n".join(exposure["row_ids"]).encode()
                    ).hexdigest(),
                    "row_ids": None,
                }
                for index, exposure in enumerate(gathered_exposure)
                if exposure is not None
            },
        }
        json_dump(args.output / "run_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
