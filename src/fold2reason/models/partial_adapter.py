"""Deterministic parameter-budget-matched partial-FT selection and I/O."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch


FORMAT_VERSION = "m3-gt-v2-partial-ft-v1"
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _layer_index(name: str) -> int | None:
    match = LAYER.search(name)
    return int(match.group(1)) if match else None


def _module_prefix(name: str) -> str:
    return name.rsplit(".", 1)[0]


def ordered_partial_groups(model: torch.nn.Module) -> list[dict[str, Any]]:
    """Return the pre-registered state/norm -> attention -> MLP group order."""

    names = [name for name, _ in model.named_parameters()]
    layers = sorted(
        {index for name in names if (index := _layer_index(name)) is not None},
        reverse=True,
    )
    groups: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(label: str, selected: list[str]) -> None:
        fresh = [name for name in selected if name not in seen]
        if not fresh:
            return
        seen.update(fresh)
        groups.append({"label": label, "parameter_names": sorted(fresh)})

    add(
        "final_norm",
        [
            name
            for name in names
            if _layer_index(name) is None and name.endswith("model.norm.weight")
        ],
    )
    state_markers = (
        ".input_layernorm.",
        ".post_attention_layernorm.",
        ".linear_attn.A_log",
        ".linear_attn.dt_bias",
        ".linear_attn.conv1d.",
        ".linear_attn.norm.",
        ".self_attn.q_norm.",
        ".self_attn.k_norm.",
    )
    for layer in layers:
        prefix = f"layers.{layer}."
        add(
            f"layer_{layer:02d}_state_norm",
            [
                name
                for name in names
                if prefix in name and any(marker in name for marker in state_markers)
            ],
        )

    attention_markers = (
        ".self_attn.q_proj.",
        ".self_attn.k_proj.",
        ".self_attn.v_proj.",
        ".self_attn.o_proj.",
        ".linear_attn.in_proj_qkv.",
        ".linear_attn.in_proj_z.",
        ".linear_attn.in_proj_b.",
        ".linear_attn.in_proj_a.",
        ".linear_attn.out_proj.",
    )
    for layer in layers:
        prefix = f"layers.{layer}."
        layer_names = [name for name in names if prefix in name]
        for marker in attention_markers:
            selected = [name for name in layer_names if marker in name]
            if selected:
                add(
                    f"layer_{layer:02d}_{_module_prefix(selected[0]).split('.')[-1]}",
                    selected,
                )

    mlp_markers = (".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.down_proj.")
    for layer in layers:
        prefix = f"layers.{layer}."
        layer_names = [name for name in names if prefix in name]
        for marker in mlp_markers:
            selected = [name for name in layer_names if marker in name]
            if selected:
                add(
                    f"layer_{layer:02d}_{_module_prefix(selected[0]).split('.')[-1]}",
                    selected,
                )
    return groups


def select_partial_parameters(
    model: torch.nn.Module,
    target_parameters: int,
    tolerance_low: float = 0.5,
    tolerance_high: float = 2.0,
) -> dict[str, Any]:
    if target_parameters <= 0:
        raise ValueError("target_parameters must be positive")
    parameters = dict(model.named_parameters())
    for parameter in parameters.values():
        parameter.requires_grad_(False)
    groups = ordered_partial_groups(model)
    candidates = []
    selected_names: list[str] = []
    count = 0
    for group_index, group in enumerate(groups):
        selected_names.extend(group["parameter_names"])
        count += sum(parameters[name].numel() for name in group["parameter_names"])
        candidates.append(
            {
                "prefix_groups": group_index + 1,
                "last_group": group["label"],
                "trainable_parameters": count,
                "absolute_budget_error": abs(count - target_parameters),
            }
        )
        if count > tolerance_high * target_parameters:
            break
    if not candidates:
        raise RuntimeError("partial-FT candidate list is empty")
    chosen = min(
        candidates,
        key=lambda row: (row["absolute_budget_error"], row["prefix_groups"]),
    )
    chosen_groups = groups[: chosen["prefix_groups"]]
    chosen_names = [
        name for group in chosen_groups for name in group["parameter_names"]
    ]
    for name in chosen_names:
        parameters[name].requires_grad_(True)
    manifest = {
        "format_version": FORMAT_VERSION,
        "selection_contract": (
            "cumulative deterministic prefix: final norm, top-down state/"
            "norm groups, top-down attention projections, top-down MLP projections"
        ),
        "target_parameters": target_parameters,
        "tolerance": [tolerance_low, tolerance_high],
        "within_tolerance": (
            tolerance_low * target_parameters
            <= chosen["trainable_parameters"]
            <= tolerance_high * target_parameters
        ),
        "chosen": chosen,
        "groups": chosen_groups,
        "trainable_parameter_names": chosen_names,
        "candidate_prefixes_considered": candidates,
    }
    return manifest


def select_full_parameters(model: torch.nn.Module) -> dict[str, Any]:
    """Select every LM parameter for the conditional 2B full-FT upper bound."""

    parameters = dict(model.named_parameters())
    names = sorted(parameters)
    for parameter in parameters.values():
        parameter.requires_grad_(True)
    count = sum(parameter.numel() for parameter in parameters.values())
    return {
        "format_version": FORMAT_VERSION,
        "selection_contract": "all language-model parameters trainable",
        "target_parameters": count,
        "tolerance": [1.0, 1.0],
        "within_tolerance": True,
        "chosen": {
            "prefix_groups": None,
            "last_group": "all_language_model_parameters",
            "trainable_parameters": count,
            "absolute_budget_error": 0,
        },
        "groups": [
            {
                "label": "all_language_model_parameters",
                "parameter_names": names,
            }
        ],
        "trainable_parameter_names": names,
        "candidate_prefixes_considered": [],
    }


def save_partial_delta(
    model: torch.nn.Module,
    output: Path,
    manifest: dict[str, Any],
) -> None:
    parameters = dict(model.named_parameters())
    state = {
        name: parameters[name].detach().cpu().contiguous()
        for name in manifest["trainable_parameter_names"]
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "manifest": manifest,
            "state_dict": state,
        },
        output / "partial_ft_delta.pt",
    )
    (output / "partial_ft_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def load_partial_delta(model: torch.nn.Module, path: Path) -> dict[str, Any]:
    payload = torch.load(
        path / "partial_ft_delta.pt", map_location="cpu", weights_only=False
    )
    if payload.get("format_version") != FORMAT_VERSION:
        raise RuntimeError("unsupported partial-FT delta format")
    parameters = dict(model.named_parameters())
    for name, value in payload["state_dict"].items():
        if name not in parameters:
            raise KeyError(f"partial-FT parameter missing in base model: {name}")
        if parameters[name].shape != value.shape:
            raise ValueError(
                f"partial-FT shape mismatch for {name}: "
                f"{tuple(value.shape)} != {tuple(parameters[name].shape)}"
            )
        parameters[name].data.copy_(
            value.to(device=parameters[name].device, dtype=parameters[name].dtype)
        )
    return payload["manifest"]
