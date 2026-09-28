#!/usr/bin/env python3
"""Distributed Phase-2 shared spatial-workspace training."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

# Long SEQ+MSA+template prompts vary from <1K to >15K tokens.  Expandable
# allocator segments prevent reserved-memory fragmentation from accumulating
# as those shapes change across an epoch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from fold2reason.evaluation.geometry import evaluate_rows
from fold2reason.models.geometry import add_all_attention_lora, load_lora, load_text_model, module_match_summary
from fold2reason.losses.geometry import geometry_loss
from fold2reason.models.workspace import WorkspaceGeometryModel
from fold2reason.training.geometry import (
    append_jsonl,
    build_epoch_groups,
    composite_score,
    environment_info,
    json_dump,
    reduce_components,
    seed_everything,
    setup_distributed,
)


ARM_CONFIG = {
    "m1": {"bridge": None, "uses_memory": False, "process": None},
    "m2": {"bridge": "real", "uses_memory": False, "process": None},
    "m3": {"bridge": "real", "uses_memory": True, "process": None},
    "m3_no_retrieval": {
        "bridge": "real",
        "uses_memory": True,
        "process": None,
        "retrieval_enabled": False,
    },
    "m3_no_geometry": {
        "bridge": "real",
        "uses_memory": True,
        "process": None,
        "geometry_head_enabled": False,
    },
    "m3_no_relation": {
        "bridge": None,
        "uses_memory": True,
        "process": None,
        "geometry_head_enabled": True,
    },
    "m4": {"bridge": "shuffled", "uses_memory": True, "process": None},
    "m5": {"bridge": "ordinary", "uses_memory": True, "process": None},
    "m3p": {
        "bridge": None,
        "uses_memory": True,
        "process": "gt",
        "student_ce": True,
        "distill": False,
        "teacher_mode": "matched",
        "base_kl": False,
        "specificity": True,
    },
    "m3d": {
        "bridge": None,
        "uses_memory": True,
        "process": "current",
        "student_ce": False,
        "distill": True,
        "teacher_mode": "matched",
        "base_kl": True,
        "specificity": True,
    },
    "m3gt": {
        "bridge": None,
        "uses_memory": True,
        "process": "gt",
        "student_ce": True,
        "distill": True,
        "teacher_mode": "matched",
        "base_kl": True,
        "specificity": True,
    },
    "m3gt_shuffle": {
        "bridge": None,
        "uses_memory": True,
        "process": "gt",
        "student_ce": True,
        "distill": True,
        "teacher_mode": "shuffled",
        "base_kl": True,
        "specificity": False,
    },
}


def parse_args() -> argparse.Namespace:
    project = Path.cwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARM_CONFIG), required=True)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3.5-9B"))
    parser.add_argument(
        "--cache", type=Path, default=project / "artifacts/cache/openfold_phase2_workspace_v0.pt"
    )
    parser.add_argument(
        "--process-cache",
        type=Path,
        default=None,
        help="Frozen M3-GT process sidecar produced by prepare_m3_gt_process_cache.py.",
    )
    parser.add_argument(
        "--frozen-decoder-checkpoint",
        type=Path,
        default=project / "outputs/phase0_causal_audit/head_only_base/seed-20260729/final",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "outputs/qwen35_9b_shared_spatial_workspace/phase2_workspace_v0/m3/seed-20260729",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--workspace-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--workspace-width", type=int, default=256)
    parser.add_argument("--memory-tokens", type=int, default=16)
    parser.add_argument("--max-sparse-pairs", type=int, default=2048)
    parser.add_argument("--retrieval-temperature", type=float, default=0.07)
    parser.add_argument("--loss-retrieval", type=float, default=1.0)
    parser.add_argument("--loss-relation", type=float, default=1.0)
    parser.add_argument("--loss-geometry", type=float, default=1.0)
    parser.add_argument("--loss-teacher-process", type=float, default=1.0)
    parser.add_argument("--loss-student-process", type=float, default=1.0)
    parser.add_argument("--loss-distill", type=float, default=1.0)
    parser.add_argument("--loss-specificity", type=float, default=0.2)
    parser.add_argument("--loss-base-kl", type=float, default=0.1)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--specificity-margin", type=float, default=0.1)
    parser.add_argument("--max-memory-dropout", type=float, default=0.75)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=0,
        help=(
            "Stop at the first optimizer-step boundary reaching this many data tokens. "
            "Teacher re-forwards do not count as new data exposure."
        ),
    )
    parser.add_argument(
        "--exposure-manifest",
        type=Path,
        default=None,
        help=(
            "Frozen fixed-token manifest containing seed_orders[seed].ordered_ids. "
            "When set, train rows follow that exact ordered exposure sequence."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260729)
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
    parser.add_argument("--fast-eval-examples", type=int, default=24)
    parser.add_argument("--final-eval-max-examples", type=int, default=0)
    parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=1,
        help="0 evaluates only at the fixed-token endpoint; otherwise evaluate every N epochs.",
    )
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument(
        "--formal-fixed-steps",
        action="store_true",
        help="Treat --max-optimizer-steps as a formal endpoint rather than a smoke stop.",
    )
    parser.add_argument(
        "--stop-after-optimizer-steps",
        type=int,
        default=0,
        help=(
            "Diagnostic early stop that does not shorten the LR schedule. "
            "Unlike --max-optimizer-steps, this preserves the full fixed-token scheduler."
        ),
    )
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help=(
            "Resume at the epoch boundary stored in a workspace checkpoint. "
            "The checkpoint must contain adapter/, workspace.pt, and trainer_state.pt."
        ),
    )
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--monitor-training-dynamics",
        action="store_true",
        help=(
            "Record per-optimizer-step loss aliases, pre-clip gradient diagnostics, "
            "and protein/relation-forward activation maxima. Disabled by default."
        ),
    )
    parser.add_argument(
        "--activation-layer-index",
        type=int,
        default=-1,
        help=(
            "0-based decoder block to monitor. The default (-1) selects the final "
            "block in the first half of the language backbone."
        ),
    )
    parser.add_argument(
        "--skip-final-evaluation",
        action="store_true",
        help="Save the final model without running validation inside the trainer.",
    )
    parser.add_argument(
        "--rank-capacity-order",
        default=os.environ.get("M3_RANK_CAPACITY_ORDER", ""),
        help=(
            "Optional comma-separated DDP ranks ordered from most to least "
            "available memory. Longer examples in each global batch are "
            "assigned to earlier ranks without changing the sample set."
        ),
    )
    parser.add_argument(
        "--activation-offload",
        action="store_true",
        default=os.environ.get("M3_ACTIVATION_OFFLOAD", "0") == "1",
        help=(
            "Store tensors saved for backward in pinned CPU memory. This keeps "
            "the computation and gradients unchanged while lowering peak GPU "
            "memory for long prompts on shared accelerators."
        ),
    )
    return parser.parse_args()


class TrainingDynamicsActivationMonitor:
    """Detach-only activation maxima for the two Fold2Space LM forwards.

    Hooks are active only during the outer model forward. Gradient-checkpoint
    recomputation happens during backward after ``end_model_forward`` and is
    therefore ignored rather than double-counted.
    """

    PATHS = ("protein_encoder", "relation_answer")

    def __init__(
        self,
        model: WorkspaceGeometryModel,
        requested_layer_index: int,
        device: torch.device,
    ) -> None:
        self.device = device
        self.active = False
        self.call_index = 0
        self.current_path: str | None = None
        self.current_attention_mask: torch.Tensor | None = None
        self.local_maxima: dict[str, torch.Tensor | None] = {
            path: None for path in self.PATHS
        }

        tower = model.text_tower()
        indexed_layers: dict[int, list[tuple[str, torch.nn.Module]]] = {}
        for name, module in tower.named_modules():
            match = re.search(r"(?:^|\.)layers\.(\d+)$", name)
            if match:
                indexed_layers.setdefault(int(match.group(1)), []).append((name, module))
        if not indexed_layers:
            raise RuntimeError("Could not locate decoder blocks under the Qwen text tower")
        layer_count = max(indexed_layers) + 1
        if sorted(indexed_layers) != list(range(layer_count)):
            raise RuntimeError(
                f"Decoder block indices are not contiguous: {sorted(indexed_layers)}"
            )
        layer_index = (
            (layer_count // 2) - 1
            if requested_layer_index < 0
            else requested_layer_index
        )
        if not 0 <= layer_index < layer_count:
            raise ValueError(
                f"activation layer {layer_index} is outside [0, {layer_count - 1}]"
            )
        relative_path, layer = min(indexed_layers[layer_index], key=lambda item: len(item[0]))
        full_paths = [
            name for name, candidate in model.named_modules() if candidate is layer
        ]
        self.layer_index = layer_index
        self.layer_count = layer_count
        self.layer_module_path = min(full_paths, key=len) if full_paths else relative_path
        self.layer_module_class = type(layer).__name__
        text_config = getattr(model.language_model.config, "text_config", None)
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            layer_types = getattr(getattr(tower, "config", None), "layer_types", None)
        self.layer_type = (
            str(layer_types[layer_index])
            if layer_types is not None and len(layer_types) > layer_index
            else self.layer_module_class
        )
        self.measurement_location = "decoder block output after residual updates"
        self.reduction = (
            "max(abs(hidden_state)) over valid tokens; maximum over microbatches "
            "then MAX over data-parallel ranks"
        )
        self._tower_pre_handle = tower.register_forward_pre_hook(
            self._tower_pre_hook, with_kwargs=True
        )
        self._layer_handle = layer.register_forward_hook(self._layer_hook)

    def begin_model_forward(self) -> None:
        if self.active:
            raise RuntimeError("Activation monitor received nested model forwards")
        self.active = True
        self.call_index = 0
        self.current_path = None
        self.current_attention_mask = None

    def end_model_forward(self, relation_expected: bool) -> None:
        expected_calls = 2 if relation_expected else 1
        if self.call_index != expected_calls:
            raise RuntimeError(
                f"Expected {expected_calls} text-tower forwards, observed {self.call_index}"
            )
        self.active = False
        self.current_path = None
        self.current_attention_mask = None

    def _tower_pre_hook(
        self,
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if not self.active:
            return
        if self.call_index >= len(self.PATHS):
            raise RuntimeError("Unexpected extra text-tower forward during monitored step")
        self.current_path = self.PATHS[self.call_index]
        self.call_index += 1
        mask = kwargs.get("attention_mask")
        if mask is None:
            input_tensor = kwargs.get("input_ids")
            if input_tensor is None:
                input_tensor = kwargs.get("inputs_embeds")
            if input_tensor is None and args:
                input_tensor = args[0]
            if input_tensor is None:
                raise RuntimeError("Cannot infer valid-token mask for monitored forward")
            mask = torch.ones(
                input_tensor.shape[:2], dtype=torch.bool, device=input_tensor.device
            )
        self.current_attention_mask = mask.detach().bool()

    def _layer_hook(
        self,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not self.active or self.current_path is None:
            return
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise RuntimeError(
                f"Monitored decoder block returned unsupported output {type(output)!r}"
            )
        mask = self.current_attention_mask
        if mask is None or tuple(mask.shape) != tuple(hidden.shape[:2]):
            raise RuntimeError(
                f"Activation mask {None if mask is None else tuple(mask.shape)} does not "
                f"match hidden states {tuple(hidden.shape)}"
            )
        value = hidden.detach().float().abs().masked_fill(~mask.unsqueeze(-1), 0.0).amax()
        previous = self.local_maxima[self.current_path]
        self.local_maxima[self.current_path] = (
            value if previous is None else torch.maximum(previous, value)
        )

    def reduce_step(self, relation_expected: bool) -> dict[str, float]:
        paths = self.PATHS if relation_expected else self.PATHS[:1]
        values = []
        for path in paths:
            value = self.local_maxima[path]
            if value is None:
                raise RuntimeError(f"No activation recorded for {path}")
            values.append(value)
        packed = torch.stack(values).to(device=self.device, dtype=torch.float32)
        dist.all_reduce(packed, op=dist.ReduceOp.MAX)
        result = {
            f"activation_abs_max_{path}": float(value)
            for path, value in zip(paths, packed.cpu().tolist())
        }
        self.local_maxima = {path: None for path in self.PATHS}
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "layer_index_0_based": self.layer_index,
            "layer_count": self.layer_count,
            "layer_module_path": self.layer_module_path,
            "layer_module_class": self.layer_module_class,
            "layer_type": self.layer_type,
            "measurement_location": self.measurement_location,
            "forward_paths": list(self.PATHS),
            "primary_figure_path": "relation_answer",
            "reduction": self.reduction,
            "padding_excluded": True,
            "checkpoint_recomputation_excluded": True,
        }


def build_padded_epoch_groups(
    rows: list[dict[str, Any]],
    world_size: int,
    seed: int,
    epoch: int,
    preserve_order: bool = False,
) -> list[tuple[list[int], list[bool]]]:
    """Build DDP groups without dropping unique examples.

    The last group is padded by repeating its final real row. Padded ranks run
    the same graph but receive zero optimization weight; active ranks are
    rescaled so DDP still averages over the real examples in that group.
    """
    if not rows:
        raise ValueError("training split is empty")
    ordered = (
        list(range(len(rows)))
        if preserve_order
        else sorted(range(len(rows)), key=lambda i: len(rows[i]["input_ids"]))
    )
    groups: list[tuple[list[int], list[bool]]] = []
    for start in range(0, len(ordered), world_size):
        indices = ordered[start : start + world_size]
        active = [True] * len(indices)
        if len(indices) < world_size:
            indices.extend([indices[-1]] * (world_size - len(indices)))
            active.extend([False] * (world_size - len(active)))
        groups.append((indices, active))
    import random

    if not preserve_order:
        random.Random(seed + epoch).shuffle(groups)
    return groups


def save_checkpoint(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output: Path,
    name: str,
    epoch: int,
    global_step: int,
    metrics: dict | None = None,
    cumulative_input_tokens: int = 0,
    elapsed_wall_seconds: float = 0.0,
) -> Path:
    checkpoint = output / "checkpoints" / name
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.module.save_pretrained(checkpoint)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            "cumulative_input_tokens": cumulative_input_tokens,
            "elapsed_wall_seconds": elapsed_wall_seconds,
        },
        checkpoint / "trainer_state.pt",
    )
    return checkpoint


def parameter_sha256(named_parameters: list[tuple[str, torch.nn.Parameter]]) -> str:
    """Hash exact initial trainable tensors for cross-arm preflight auditing."""
    digest = hashlib.sha256()
    for name, parameter in sorted(named_parameters):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    checkpoint_steps = {
        int(value)
        for value in args.checkpoint_steps.split(",")
        if value.strip()
    }
    if any(step < 0 for step in checkpoint_steps):
        raise ValueError("--checkpoint-steps must contain non-negative integers")
    if args.formal_fixed_steps and not args.max_optimizer_steps:
        raise ValueError("--formal-fixed-steps requires --max-optimizer-steps")
    arm = ARM_CONFIG[args.arm]
    rank, local_rank, world_size, device = setup_distributed()
    if args.rank_capacity_order:
        rank_capacity_order = [int(value) for value in args.rank_capacity_order.split(",")]
        if sorted(rank_capacity_order) != list(range(world_size)):
            raise ValueError(
                "--rank-capacity-order must be a permutation of all DDP ranks; "
                f"got {rank_capacity_order} for world_size={world_size}"
            )
    else:
        rank_capacity_order = list(range(world_size))
    seed_everything(args.seed, rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    log_path = args.output / "training_log.jsonl"

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    if cache["stats"].get("workspace_version") != "phase2_workspace_v0":
        raise RuntimeError("Phase-2 workspace cache required")
    process_cache = None
    if arm["process"] == "gt":
        if args.process_cache is None:
            raise ValueError(f"--process-cache is required for arm {args.arm}")
        process_cache = torch.load(args.process_cache, map_location="cpu", weights_only=False)
        if process_cache.get("stats", {}).get("generator_version") != "m3-gt-folding-process-v1":
            raise RuntimeError("M3-GT folding-process cache v1 required")
    train_rows = cache["splits"]["train"]
    exposure_contract = None
    if args.exposure_manifest is not None:
        exposure_contract = json.loads(args.exposure_manifest.read_text())
        seed_contract = exposure_contract.get("seed_orders", {}).get(str(args.seed))
        if seed_contract is None:
            raise KeyError(
                f"Exposure manifest has no frozen order for seed {args.seed}"
            )
        row_by_id = {row["id"]: row for row in train_rows}
        if len(row_by_id) != len(train_rows):
            raise RuntimeError("Canonical cache contains duplicate unique train IDs")
        ordered_ids = seed_contract["ordered_ids"]
        missing = sorted(set(ordered_ids) - set(row_by_id))
        if missing:
            raise RuntimeError(f"Exposure manifest contains missing IDs: {missing[:5]}")
        train_rows = [row_by_id[row_id] for row_id in ordered_ids]
        ordered_sha = hashlib.sha256(
            json.dumps(
                ordered_ids, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if ordered_sha != seed_contract["ordered_ids_sha256"]:
            raise RuntimeError("Exposure order SHA256 mismatch")
        if args.epochs != 1:
            raise ValueError("Exposure-manifest runs must use exactly one epoch")
        if args.max_input_tokens:
            raise ValueError(
                "Exposure-manifest and online --max-input-tokens stopping are mutually exclusive"
            )
    if args.overfit_samples:
        train_rows = sorted(train_rows, key=lambda row: len(row["input_ids"]))[: args.overfit_samples]
    unique_train_examples = len(train_rows)
    ddp_padding_examples_per_epoch = (-unique_train_examples) % world_size
    if process_cache is not None:
        missing_process = [
            row["id"] for row in train_rows if row["id"] not in process_cache["splits"]["train"]
        ]
        if missing_process:
            raise RuntimeError(
                f"process cache is missing {len(missing_process)} training rows: "
                f"{missing_process[:5]}"
            )

    base, loading_info = load_text_model(args.model, device)
    if args.resume_from_checkpoint is None:
        lora = add_all_attention_lora(
            base,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            gradient_checkpointing=not args.disable_gradient_checkpointing,
        )
    else:
        for required in (
            args.resume_from_checkpoint / "adapter/adapter_config.json",
            args.resume_from_checkpoint / "workspace.pt",
            args.resume_from_checkpoint / "trainer_state.pt",
        ):
            if not required.is_file():
                raise FileNotFoundError(f"resume checkpoint is incomplete: {required}")
        lora = load_lora(
            base,
            args.resume_from_checkpoint / "adapter",
            trainable=True,
            gradient_checkpointing=not args.disable_gradient_checkpointing,
        )
    lora_audit = module_match_summary(lora)
    model = WorkspaceGeometryModel(
        lora,
        args.frozen_decoder_checkpoint,
        workspace_width=args.workspace_width,
        memory_tokens=args.memory_tokens,
        fingerprint_dim=cache["stats"]["teacher_fingerprint_dim"],
        max_sparse_pairs=args.max_sparse_pairs,
        relation_uses_memory=arm["uses_memory"],
        retrieval_temperature=args.retrieval_temperature,
        geometry_head_enabled=arm.get("geometry_head_enabled", True),
    )
    retrieval_enabled = arm.get("retrieval_enabled", True)
    initial_retrieval_projection = (
        model.workspace.retrieval_projection.weight.detach().cpu().clone()
    )
    if not retrieval_enabled:
        for parameter in model.workspace.retrieval_projection.parameters():
            parameter.requires_grad_(False)
    if model.geometry_head_enabled:
        model.coordinate_head.to(device)
        model.distogram_head.to(device)
    model.workspace.to(device)
    if args.resume_from_checkpoint is not None:
        workspace_state = torch.load(
            args.resume_from_checkpoint / "workspace.pt",
            map_location="cpu",
            weights_only=False,
        )
        model.load_workspace_state_dict(workspace_state)
    lora_parameters = [parameter for parameter in model.language_model.parameters() if parameter.requires_grad]
    workspace_parameters = [parameter for parameter in model.workspace.parameters() if parameter.requires_grad]
    retrieval_projection_trainable_parameters = sum(
        parameter.numel()
        for parameter in model.workspace.retrieval_projection.parameters()
        if parameter.requires_grad
    )
    initial_trainable_sha256 = None
    if os.environ.get("M3_AUDIT_INITIAL_PARAMETER_HASH") == "1":
        initial_trainable_sha256 = parameter_sha256(
            [
                (f"language_model.{name}", parameter)
                for name, parameter in model.language_model.named_parameters()
                if parameter.requires_grad
            ]
            + [
                (f"workspace.{name}", parameter)
                for name, parameter in model.workspace.named_parameters()
                if parameter.requires_grad
            ]
        )
    frozen_decoder_parameters = (
        list(model.coordinate_head.parameters())
        + list(model.distogram_head.parameters())
        if model.geometry_head_enabled
        else []
    )
    if any(parameter.requires_grad for parameter in frozen_decoder_parameters):
        raise RuntimeError("Frozen decoder contains trainable parameters")

    activation_monitor = (
        TrainingDynamicsActivationMonitor(
            model,
            requested_layer_index=args.activation_layer_index,
            device=device,
        )
        if args.monitor_training_dynamics
        else None
    )

    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": args.lora_learning_rate, "weight_decay": args.weight_decay},
            {"params": workspace_parameters, "lr": args.workspace_learning_rate, "weight_decay": args.weight_decay},
        ],
        fused=True,
    )
    groups_per_epoch = math.ceil(len(train_rows) / world_size)
    steps_per_epoch = math.ceil(groups_per_epoch / args.gradient_accumulation)
    effective_epochs = args.epochs
    tokens_per_epoch = 0
    for row in train_rows:
        row_tokens = len(row["input_ids"])
        if arm["process"] == "current":
            row_tokens += len(row["bridge"]["real"]["input_ids"])
        elif arm["process"] == "gt":
            row_tokens += len(process_cache["splits"]["train"][row["id"]]["input_ids"])
        elif arm["bridge"] is not None:
            row_tokens += len(row["bridge"][arm["bridge"]]["input_ids"])
        tokens_per_epoch += row_tokens
    if args.max_input_tokens:
        effective_epochs = max(
            args.epochs,
            math.ceil(args.max_input_tokens / max(tokens_per_epoch, 1)) + 1,
        )
    scheduled_steps = steps_per_epoch * effective_epochs
    if args.max_input_tokens:
        scheduled_steps = min(
            scheduled_steps,
            math.ceil(args.max_input_tokens / max(tokens_per_epoch, 1) * steps_per_epoch) + 2,
        )
    if args.max_optimizer_steps:
        scheduled_steps = min(scheduled_steps, args.max_optimizer_steps)
    warmup_steps = max(1, round(scheduled_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, scheduled_steps)

    resume_state = None
    start_epoch = 0
    global_step = 0
    cumulative_tokens = 0
    elapsed_offset = 0.0
    best_score = -float("inf")
    best_checkpoint = None
    if args.resume_from_checkpoint is not None:
        resume_state = torch.load(
            args.resume_from_checkpoint / "trainer_state.pt",
            map_location=device,
            weights_only=False,
        )
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch = int(resume_state["epoch"]) + 1
        global_step = int(resume_state["global_step"])
        cumulative_tokens = int(resume_state.get("cumulative_input_tokens", 0))
        elapsed_offset = float(resume_state.get("elapsed_wall_seconds", 0.0))
        if start_epoch > args.epochs:
            raise ValueError(
                f"resume checkpoint epoch {start_epoch} exceeds requested epochs {args.epochs}"
            )
        if resume_state.get("metrics"):
            best_score = composite_score(resume_state["metrics"])
        existing_best = args.output / "checkpoints/best"
        if (existing_best / "trainer_state.pt").is_file():
            best_state = torch.load(
                existing_best / "trainer_state.pt",
                map_location="cpu",
                weights_only=False,
            )
            if best_state.get("metrics"):
                existing_score = composite_score(best_state["metrics"])
                if existing_score >= best_score:
                    best_score = existing_score
                    best_checkpoint = existing_best

        if rank == 0 and log_path.exists():
            original_lines = log_path.read_text().splitlines()
            kept_lines: list[str] = []
            last_train_record: dict | None = None
            for line in original_lines:
                record = json.loads(line)
                record_step = int(record.get("global_step", 0))
                if record_step <= global_step:
                    kept_lines.append(line)
                    if record.get("event") == "train_step":
                        last_train_record = record
            backup = log_path.with_name(
                f"training_log.pre_resume_step{global_step}.jsonl"
            )
            if not backup.exists():
                backup.write_text("\n".join(original_lines) + "\n")
            log_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))
            if last_train_record is not None and not cumulative_tokens:
                cumulative_tokens = int(last_train_record.get("cumulative_input_tokens", 0))
                elapsed_offset = float(last_train_record.get("wall_seconds", 0.0))

    if rank == 0:
        if log_path.exists() and args.resume_from_checkpoint is None:
            log_path.unlink()
        json_dump(
            args.output / "run_config.json",
            {
                "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "arm_config": arm,
                "environment": environment_info(),
                "world_size": world_size,
                "cache_stats": cache["stats"],
                "model_loading": loading_info,
                "lora_audit": lora_audit,
                "lora_trainable_parameters": sum(p.numel() for p in lora_parameters),
                "workspace_trainable_parameters": sum(p.numel() for p in workspace_parameters),
                "frozen_decoder_parameters": sum(p.numel() for p in frozen_decoder_parameters),
                "frozen_decoder_trainable_parameters": sum(p.numel() for p in frozen_decoder_parameters if p.requires_grad),
                "training_dynamics": {
                    "enabled": activation_monitor is not None,
                    "optimizer_step_axis": True,
                    "gradient_scaler_enabled": False,
                    "gradient_norm": {
                        "parameters": "all trainable LoRA and workspace parameters",
                        "timing": (
                            "after gradient accumulation and DDP synchronization; "
                            "before clipping and optimizer.step"
                        ),
                        "global_norm": "L2",
                        "clip_threshold": args.max_grad_norm,
                    },
                    "loss_statistics": {
                        "optimizer_step": (
                            "equal-weight mean over the real proteins in the optimizer "
                            "step after DDP SUM reduction"
                        ),
                        "relation_ce": (
                            "per-protein mean cross entropy over supervised answer tokens, "
                            "then equal-weight mean over proteins"
                        ),
                        "geometry": (
                            "weighted sum of the geometry terms after each term's native "
                            "residue/pair reduction, then equal-weight mean over proteins"
                        ),
                    },
                    "branch_weights": {
                        "relation": args.loss_relation,
                        "geometry": args.loss_geometry,
                        "retrieval": args.loss_retrieval,
                    },
                    "geometry_component_weights": {
                        "aligned_coord": 1.0,
                        "pair_distance": 1.0,
                        "contact": 0.5,
                        "distogram": 0.3,
                        "backbone_local": 0.2,
                        "torsion": 0.2,
                        "radius_of_gyration": 0.1,
                        "input_contrastive": 0.0,
                    },
                    "activation": (
                        activation_monitor.metadata()
                        if activation_monitor is not None
                        else {"enabled": False}
                    ),
                },
                "initial_trainable_sha256": initial_trainable_sha256,
                "ordered_train_ids_sha256": hashlib.sha256(
                    json.dumps(
                        [row["id"] for row in train_rows],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
                "unique_train_examples": unique_train_examples,
                "ddp_padding_examples_per_epoch": ddp_padding_examples_per_epoch,
                "ablation_contract": {
                    "geometry_forward_enabled": model.module.geometry_head_enabled,
                    "geometry_head_loaded": model.module.geometry_head_enabled,
                    "geometry_loss_weight_effective": (
                        args.loss_geometry if model.module.geometry_head_enabled else 0.0
                    ),
                    "geometry_train_examples": (
                        len(train_rows) * effective_epochs
                        if model.module.geometry_head_enabled
                        else 0
                    ),
                    "retrieval_forward_enabled": retrieval_enabled,
                    "retrieval_loss_weight_effective": (
                        args.loss_retrieval if retrieval_enabled else 0.0
                    ),
                    "retrieval_train_examples": (
                        len(train_rows) * effective_epochs if retrieval_enabled else 0
                    ),
                    "retrieval_projection_trainable_parameters": (
                        retrieval_projection_trainable_parameters
                    ),
                    "relation_forward_enabled": arm["bridge"] is not None,
                    "relation_loss_weight_effective": (
                        args.loss_relation if arm["bridge"] is not None else 0.0
                    ),
                },
                "steps_per_epoch": steps_per_epoch,
                "effective_epochs": effective_epochs,
                "scheduled_steps": scheduled_steps,
                "warmup_steps": warmup_steps,
                "fixed_token_contract": {
                    "max_input_tokens": args.max_input_tokens,
                    "tokens_per_full_epoch": tokens_per_epoch,
                    "teacher_reforwards_count_as_data_exposure": False,
                    "exposure_manifest": (
                        str(args.exposure_manifest)
                        if args.exposure_manifest is not None
                        else None
                    ),
                    "exposure_manifest_status": (
                        exposure_contract.get("status")
                        if exposure_contract is not None
                        else None
                    ),
                    "ordered_exposure_rows": (
                        len(train_rows) if exposure_contract is not None else None
                    ),
                },
                "resume": (
                    {
                        "checkpoint": str(args.resume_from_checkpoint),
                        "start_epoch": start_epoch,
                        "global_step": global_step,
                        "cumulative_input_tokens": cumulative_tokens,
                    }
                    if args.resume_from_checkpoint is not None
                    else None
                ),
            },
        )
    dist.barrier()

    if args.save_step_zero and args.resume_from_checkpoint is not None:
        raise ValueError("--save-step-zero cannot be used when resuming")
    if 0 in checkpoint_steps and not args.save_step_zero:
        raise ValueError("checkpoint step 0 requires --save-step-zero")
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
                cumulative_input_tokens=0,
                elapsed_wall_seconds=0.0,
            )
        dist.barrier()

    loss_weights = {
        "aligned_coord": 1.0,
        "pair_distance": 1.0,
        "contact": 0.5,
        "distogram": 0.3,
        "backbone_local": 0.2,
        "torsion": 0.2,
        "radius_of_gyration": 0.1,
        "input_contrastive": 0.0,
    }
    optimizer.zero_grad(set_to_none=True)
    model.train()
    started = time.perf_counter() - elapsed_offset
    stop = False
    relation_supervised_tokens_rank = 0

    for epoch in range(start_epoch, effective_epochs):
        groups = build_padded_epoch_groups(
            train_rows,
            world_size,
            args.seed,
            epoch,
            preserve_order=exposure_contract is not None,
        )
        pending = 0
        accumulated: dict[str, float] = {}
        accumulated_microsteps = 0
        for group_index, (group, active_flags) in enumerate(groups):
            accumulation_window_start = (
                group_index // args.gradient_accumulation
            ) * args.gradient_accumulation
            accumulation_window_end = min(
                accumulation_window_start + args.gradient_accumulation,
                len(groups),
            )
            accumulation_window_examples = sum(
                sum(window_active)
                for _, window_active in groups[
                    accumulation_window_start:accumulation_window_end
                ]
            )
            if args.rank_capacity_order:
                by_length = sorted(
                    zip(group, active_flags),
                    key=lambda item: len(train_rows[item[0]]["input_ids"]),
                    reverse=True,
                )
                assigned = [0] * world_size
                assigned_active = [False] * world_size
                for (row_index, is_active), target_rank in zip(by_length, rank_capacity_order):
                    assigned[target_rank] = row_index
                    assigned_active[target_rank] = is_active
                group = assigned
                active_flags = assigned_active
            row = train_rows[group[rank]]
            is_active = active_flags[rank]
            sample_scale = (
                world_size / accumulation_window_examples if is_active else 0.0
            )
            pending += 1
            is_last = group_index == len(groups) - 1
            should_sync = pending == args.gradient_accumulation or is_last
            sync_context = contextlib.nullcontext() if should_sync else model.no_sync()
            input_ids = row["input_ids"].to(device=device, dtype=torch.long)
            marker_positions = row["marker_positions"].to(device=device, dtype=torch.long)
            candidate_fingerprints = (
                row["workspace"]["candidate_fingerprints"].to(device=device)
                if retrieval_enabled
                else None
            )
            relation_input_ids = relation_labels = None
            process_input_ids = process_labels = None
            if arm["bridge"] is not None:
                bridge = row["bridge"][arm["bridge"]]
                relation_supervised_tokens_rank += int(is_active) * int(
                    (bridge["labels"] != -100).sum().item()
                )
                relation_input_ids = bridge["input_ids"].to(device=device, dtype=torch.long)
                relation_labels = bridge["labels"].to(device=device, dtype=torch.long)
            elif arm["process"] == "current":
                bridge = row["bridge"]["real"]
                process_input_ids = bridge["input_ids"].to(device=device, dtype=torch.long)
                process_labels = bridge["labels"].to(device=device, dtype=torch.long)
            elif arm["process"] == "gt":
                process = process_cache["splits"]["train"][row["id"]]
                process_input_ids = process["input_ids"].to(device=device, dtype=torch.long)
                process_labels = process["labels"].to(device=device, dtype=torch.long)
            process_config = None
            if process_input_ids is not None:
                token_progress = (
                    min(cumulative_tokens / args.max_input_tokens, 1.0)
                    if args.max_input_tokens
                    else min(global_step / max(scheduled_steps, 1), 1.0)
                )
                process_config = {
                    "input_ids": process_input_ids,
                    "labels": process_labels,
                    "student_ce": arm["student_ce"],
                    "distill": arm["distill"],
                    "teacher_mode": arm["teacher_mode"],
                    "memory_dropout_probability": args.max_memory_dropout * token_progress,
                    "distill_temperature": args.distill_temperature,
                    "base_kl": arm["base_kl"],
                    "specificity_margin": (
                        args.specificity_margin if arm["specificity"] else None
                    ),
                }
            with sync_context:
                activation_context = (
                    torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
                    if args.activation_offload
                    else contextlib.nullcontext()
                )
                with activation_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        if activation_monitor is not None:
                            activation_monitor.begin_model_forward()
                        try:
                            outputs = model(
                                input_ids,
                                marker_positions,
                                relation_input_ids=relation_input_ids,
                                relation_labels=relation_labels,
                                candidate_fingerprints=candidate_fingerprints,
                                process_config=process_config,
                            )
                        finally:
                            if activation_monitor is not None:
                                activation_monitor.end_model_forward(
                                    relation_expected=relation_input_ids is not None
                                )
                    if model.module.geometry_head_enabled:
                        with torch.autocast(device_type="cuda", enabled=False):
                            geometry_total, components = geometry_loss(
                                outputs, row, device, loss_weights=loss_weights
                            )
                    else:
                        geometry_total = outputs["retrieval_loss"].new_zeros(())
                        components = {}
                    retrieval_loss = outputs.get("retrieval_loss")
                    if retrieval_loss is None:
                        retrieval_loss = outputs["memory_tokens"].new_zeros(())
                    total = (
                        args.loss_retrieval * retrieval_loss
                        if retrieval_enabled
                        else retrieval_loss
                    )
                    if model.module.geometry_head_enabled:
                        total = total + args.loss_geometry * geometry_total
                    retrieval_correct = (
                        (outputs["retrieval_logits"].argmax() == 0).float()
                        if "retrieval_logits" in outputs
                        else outputs["memory_tokens"].new_zeros(())
                    )
                    components = {
                        **components,
                        "geometry_total": geometry_total,
                        "retrieval_32": retrieval_loss,
                        "retrieval_correct": retrieval_correct,
                        "memory_token_norm": outputs["memory_tokens"].float().norm(dim=-1).mean(),
                    }
                    if relation_input_ids is not None:
                        total = total + args.loss_relation * outputs["relation_loss"]
                        components["relation_ce"] = outputs["relation_loss"]
                    if process_input_ids is not None:
                        total = total + args.loss_teacher_process * outputs["teacher_process_ce"]
                        components["teacher_process_ce"] = outputs["teacher_process_ce"]
                        components["process_answer_tokens"] = outputs["process_answer_tokens"]
                        components["memory_dropout_probability"] = torch.tensor(
                            process_config["memory_dropout_probability"],
                            device=device,
                        )
                        if "student_process_ce" in outputs:
                            total = total + args.loss_student_process * outputs["student_process_ce"]
                            components["student_process_ce"] = outputs["student_process_ce"]
                        if "distill_kl" in outputs:
                            total = total + args.loss_distill * outputs["distill_kl"]
                            components["distill_kl"] = outputs["distill_kl"]
                        if "specificity_margin" in outputs:
                            total = total + args.loss_specificity * outputs["specificity_margin"]
                            components["specificity_margin"] = outputs["specificity_margin"]
                            components["shuffled_teacher_ce"] = outputs["shuffled_teacher_ce"]
                        if "base_kl" in outputs:
                            total = total + args.loss_base_kl * outputs["base_kl"]
                            components["base_kl"] = outputs["base_kl"]
                    components["total"] = total
                    optimization_total = total * sample_scale
                    scaled_components = {
                        name: value * sample_scale for name, value in components.items()
                    }
                # The first backward pass for a new sequence length can trigger
                # Triton autotuning in Qwen3.5's recurrent blocks.  On shared
                # 80-GiB GPUs the allocator cache left by the forward pass can
                # otherwise starve the temporary benchmark kernels even though
                # the selected kernel and the steady-state step fit.  This is an
                # opt-in launch-time mitigation and does not change computation.
                if os.environ.get("M3_EMPTY_CACHE_BEFORE_BACKWARD") == "1":
                    torch.cuda.empty_cache()
                optimization_total.backward()
            reduced = reduce_components(scaled_components, world_size)
            for name, value in reduced.items():
                accumulated[name] = accumulated.get(name, 0.0) + value
            accumulated_microsteps += 1
            tokens = torch.tensor(
                int(is_active)
                * (
                    len(input_ids)
                    + (len(relation_input_ids) if relation_input_ids is not None else 0)
                    + (len(process_input_ids) if process_input_ids is not None else 0)
                ),
                device=device,
                dtype=torch.long,
            )
            dist.all_reduce(tokens, op=dist.ReduceOp.SUM)
            cumulative_tokens += int(tokens)
            if not should_sync:
                continue

            activation_metrics = (
                activation_monitor.reduce_step(
                    relation_expected=relation_input_ids is not None
                )
                if activation_monitor is not None
                else {}
            )
            lora_learning_rate_used = float(optimizer.param_groups[0]["lr"])
            workspace_learning_rate_used = float(optimizer.param_groups[1]["lr"])
            grad_norm = torch.nn.utils.clip_grad_norm_(
                lora_parameters + workspace_parameters, args.max_grad_norm
            )
            preclip_grad_norm = float(grad_norm)
            gradient_is_finite = math.isfinite(preclip_grad_norm)
            clipping_triggered = bool(preclip_grad_norm > args.max_grad_norm)
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
                **accumulated,
                "loss_total": accumulated["total"],
                "loss_relation": accumulated.get("relation_ce", 0.0),
                "loss_geometry": accumulated.get("geometry_total", 0.0),
                "loss_retrieval": accumulated.get("retrieval_32", 0.0),
                "weight_relation": args.loss_relation,
                "weight_geometry": args.loss_geometry,
                "weight_retrieval": args.loss_retrieval,
                "optimizer_step_microbatches": accumulated_microsteps,
                "optimizer_step_examples": accumulation_window_examples,
                "loss_is_finite": math.isfinite(accumulated["total"]),
                "lora_learning_rate_used": lora_learning_rate_used,
                "workspace_learning_rate_used": workspace_learning_rate_used,
                "lora_learning_rate": scheduler.get_last_lr()[0],
                "workspace_learning_rate": scheduler.get_last_lr()[1],
                "grad_norm": preclip_grad_norm,
                "preclip_grad_norm": preclip_grad_norm,
                "gradient_is_finite": gradient_is_finite,
                "gradient_clip_threshold": args.max_grad_norm,
                "gradient_clipping_triggered": clipping_triggered,
                "optimizer_step_skipped": False,
                **activation_metrics,
                "cumulative_input_tokens": cumulative_tokens,
                "aggregate_input_tokens_per_second": cumulative_tokens / elapsed,
                "max_memory_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
                "wall_seconds": elapsed,
            }
            if rank == 0:
                append_jsonl(log_path, record)
                print(
                    f"[workspace-train] {args.arm} step={global_step}/{scheduled_steps} "
                    f"total={record['total']:.4f} retrieval={record['retrieval_32']:.4f} "
                    f"r1={record['retrieval_correct']:.3f} mem={record['max_memory_allocated_gib']:.1f}GiB",
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
                        cumulative_input_tokens=cumulative_tokens,
                        elapsed_wall_seconds=elapsed,
                    )
                dist.barrier()
            if args.max_optimizer_steps and global_step >= args.max_optimizer_steps:
                stop = True
                break
            if (
                args.stop_after_optimizer_steps
                and global_step >= args.stop_after_optimizer_steps
            ):
                stop = True
                break
            if args.max_input_tokens and cumulative_tokens >= args.max_input_tokens:
                stop = True
                break

        should_evaluate_epoch = (not args.skip_final_evaluation) and (
            stop
            or (
                args.eval_every_epochs > 0
                and (epoch + 1) % args.eval_every_epochs == 0
            )
        )
        metrics = None
        score = None
        if should_evaluate_epoch and model.module.geometry_head_enabled:
            diagnostic_run = bool(
                args.max_optimizer_steps or args.stop_after_optimizer_steps
            )
            if args.overfit_samples:
                epoch_eval_max_examples = min(len(train_rows), 24)
            elif diagnostic_run:
                epoch_eval_max_examples = args.final_eval_max_examples
            else:
                epoch_eval_max_examples = 0
            metrics = evaluate_rows(
                model,
                train_rows if args.overfit_samples else cache["splits"]["validation"],
                f"validation-epoch-{epoch + 1}",
                rank,
                world_size,
                device,
                max_examples=epoch_eval_max_examples,
            )
            score = composite_score(metrics)
        save_epoch_checkpoint = (
            (args.save_every_epochs > 0 and (epoch + 1) % args.save_every_epochs == 0)
            or stop
            or epoch + 1 == effective_epochs
        )
        if rank == 0 and save_epoch_checkpoint:
            if metrics is not None:
                append_jsonl(log_path, {"event": "epoch_evaluation", "epoch": epoch, "global_step": global_step, "composite_score": score, **metrics})
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                args.output,
                f"epoch-{epoch + 1:02d}",
                epoch,
                global_step,
                metrics,
                cumulative_input_tokens=cumulative_tokens,
                elapsed_wall_seconds=elapsed,
            )
            if metrics is not None and score > best_score:
                best_score = score
                best_checkpoint = save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    args.output,
                    "best",
                    epoch,
                    global_step,
                    metrics,
                    cumulative_input_tokens=cumulative_tokens,
                    elapsed_wall_seconds=elapsed,
                )
            if metrics is not None:
                print(f"[workspace-epoch] {args.arm} epoch={epoch + 1} TM={metrics['tm_score']:.4f} contact={metrics['contact_f1']:.4f}", flush=True)
        model.train()
        if stop:
            break

    dist.barrier()
    if rank == 0:
        model.module.save_pretrained(args.output / "final")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        tokenizer.save_pretrained(args.output / "final/adapter")
    dist.barrier()
    evaluations = {}
    if model.module.geometry_head_enabled and not args.skip_final_evaluation:
        for split in (
            name
            for name in ("validation", "validation_rare")
            if name in cache["splits"]
        ):
            evaluations[split] = evaluate_rows(
                model,
                cache["splits"][split],
                split,
                rank,
                world_size,
                device,
                max_examples=args.final_eval_max_examples,
            )
    if rank == 0:
        retrieval_projection_update_norm = float(
            (
                model.module.workspace.retrieval_projection.weight.detach().cpu()
                - initial_retrieval_projection
            )
            .float()
            .norm()
            .item()
        )
        missing_checkpoint_steps = sorted(
            step
            for step in checkpoint_steps
            if not (args.output / "checkpoints" / f"step-{step:04d}" / "trainer_state.pt").is_file()
        )
        if missing_checkpoint_steps:
            raise RuntimeError(
                f"requested checkpoint steps were not saved: {missing_checkpoint_steps}"
            )
        summary = {
            "status": (
                "SMOKE_COMPLETE"
                if args.stop_after_optimizer_steps
                or (args.max_optimizer_steps and not args.formal_fixed_steps)
                else "TRAINING_COMPLETE"
            ),
            "arm": args.arm,
            "global_steps": global_step,
            "cumulative_input_tokens": cumulative_tokens,
            "target_input_tokens": args.max_input_tokens or None,
            "unique_train_examples": unique_train_examples,
            "ddp_padding_examples_per_epoch": ddp_padding_examples_per_epoch,
            "input_token_overshoot": (
                cumulative_tokens - args.max_input_tokens if args.max_input_tokens else None
            ),
            "equivalent_full_data_epochs": (
                cumulative_tokens / tokens_per_epoch if tokens_per_epoch else None
            ),
            "training_wall_seconds": time.perf_counter() - started,
            "best_composite_score": best_score,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
            "final_checkpoint": str(args.output / "final"),
            "evaluations": evaluations,
            "max_memory_allocated_gib_rank0": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
            "ablation_audit": {
                "geometry_forward_enabled": model.module.geometry_head_enabled,
                "geometry_head_enabled": model.module.geometry_head_enabled,
                "geometry_forward_calls_rank0": model.module.geometry_forward_calls,
                "geometry_loss_weight_effective": (
                    args.loss_geometry if model.module.geometry_head_enabled else 0.0
                ),
                "geometry_train_examples": (
                    len(train_rows) * effective_epochs
                    if model.module.geometry_head_enabled
                    else 0
                ),
                "retrieval_forward_enabled": retrieval_enabled,
                "retrieval_loss_weight_effective": (
                    args.loss_retrieval if retrieval_enabled else 0.0
                ),
                "retrieval_train_examples": (
                    len(train_rows) * effective_epochs if retrieval_enabled else 0
                ),
                "retrieval_projection_trainable_parameters": (
                    retrieval_projection_trainable_parameters
                ),
                "retrieval_projection_update_norm": retrieval_projection_update_norm,
                "relation_forward_enabled": arm["bridge"] is not None,
                "relation_loss_weight_effective": (
                    args.loss_relation if arm["bridge"] is not None else 0.0
                ),
                "relation_forward_calls_rank0": model.module.relation_forward_calls,
                "relation_supervised_tokens_rank0": relation_supervised_tokens_rank,
            },
        }
        json_dump(args.output / "run_summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
