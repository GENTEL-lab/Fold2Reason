#!/usr/bin/env python3
"""Qwen3.5 LoRA text tower with residue-coordinate and distogram heads."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from transformers import AutoConfig, Qwen3_5ForCausalLM


COORDINATE_SCALE = 20.0
DEFAULT_COORDINATE_MODE = "free"
ALL_ATTN_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

BOND_N_CA = 1.458
BOND_CA_C = 1.525
BOND_C_N = 1.329
BOND_C_O = 1.231
ANGLE_N_CA_C = math.radians(111.2)
ANGLE_CA_C_N = math.radians(116.2)
ANGLE_C_N_CA = math.radians(121.7)
ANGLE_CA_C_O = math.radians(120.8)
O_DIHEDRAL = math.pi


def load_text_model(
    model_path: Path,
    device: torch.device,
) -> tuple[Qwen3_5ForCausalLM, dict[str, Any]]:
    config = AutoConfig.from_pretrained(
        model_path, local_files_only=True
    ).text_config
    config._attn_implementation = "sdpa"
    config.use_cache = False
    model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
        model_path,
        config=config,
        key_mapping={r"^model\.language_model\.": "model."},
        dtype=torch.bfloat16,
        device_map={"": device.index},
        local_files_only=True,
        output_loading_info=True,
        low_cpu_mem_usage=True,
    )
    missing = sorted(loading_info["missing_keys"])
    unexpected = sorted(loading_info["unexpected_keys"])
    if missing or unexpected:
        raise RuntimeError(
            f"Text-only checkpoint load mismatch: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    model.config.use_cache = False
    return model, {
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def add_all_attention_lora(
    base: Qwen3_5ForCausalLM,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    gradient_checkpointing: bool = True,
) -> PeftModel:
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=ALL_ATTN_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(base, config)
    model.enable_input_require_grads()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def load_lora(
    base: Qwen3_5ForCausalLM,
    adapter_path: Path,
    trainable: bool,
    gradient_checkpointing: bool = False,
) -> PeftModel:
    model = PeftModel.from_pretrained(
        base, adapter_path, is_trainable=trainable
    )
    if trainable:
        model.enable_input_require_grads()
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
    return model


class CoordinateHead(nn.Module):
    def __init__(self, hidden_size: int = 4096, width: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj_in = nn.Linear(hidden_size, width)
        self.activation = nn.SiLU()
        self.proj_out = nn.Linear(width, 12)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        coords = self.proj_out(
            self.activation(self.proj_in(self.norm(hidden.float())))
        ).reshape(-1, 4, 3)
        coords = coords * COORDINATE_SCALE
        ca_center = coords[:, 1].mean(dim=0, keepdim=True)
        return coords - ca_center[None, :, :]


def place_atom(
    atom_a: torch.Tensor,
    atom_b: torch.Tensor,
    atom_c: torch.Tensor,
    bond_length: float,
    bond_angle: float,
    dihedral: torch.Tensor,
) -> torch.Tensor:
    bc = F.normalize(atom_c - atom_b, dim=-1, eps=1e-6)
    normal = F.normalize(torch.cross(atom_b - atom_a, bc, dim=-1), dim=-1, eps=1e-6)
    in_plane = torch.cross(normal, bc, dim=-1)
    return atom_c + bond_length * (
        -math.cos(bond_angle) * bc
        + math.sin(bond_angle)
        * (torch.cos(dihedral) * in_plane + torch.sin(dihedral) * normal)
    )


class InternalCoordinateHead(nn.Module):
    def __init__(self, hidden_size: int = 4096, width: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj_in = nn.Linear(hidden_size, width)
        self.activation = nn.SiLU()
        self.proj_out = nn.Linear(width, 6)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        raw = self.proj_out(
            self.activation(self.proj_in(self.norm(hidden.float())))
        ).reshape(-1, 3, 2)
        defaults = torch.tensor(
            [[[0.0, 1.0], [0.0, 1.0], [0.0, -1.0]]],
            dtype=raw.dtype,
            device=raw.device,
        )
        torsion_sincos = F.normalize(defaults + raw, dim=-1, eps=1e-6)
        torsion_angles = torch.atan2(
            torsion_sincos[..., 0], torsion_sincos[..., 1]
        )
        return self.build_backbone(torsion_angles)

    def build_backbone(self, torsion_angles: torch.Tensor) -> torch.Tensor:
        length = torsion_angles.shape[0]
        device = torsion_angles.device
        dtype = torsion_angles.dtype
        n_atom = torch.tensor([-BOND_N_CA, 0.0, 0.0], device=device, dtype=dtype)
        ca_atom = torch.zeros(3, device=device, dtype=dtype)
        c_atom = torch.tensor(
            [
                BOND_CA_C * math.cos(math.pi - ANGLE_N_CA_C),
                BOND_CA_C * math.sin(math.pi - ANGLE_N_CA_C),
                0.0,
            ],
            device=device,
            dtype=dtype,
        )
        residues = []
        for index in range(length):
            o_atom = place_atom(
                n_atom,
                ca_atom,
                c_atom,
                BOND_C_O,
                ANGLE_CA_C_O,
                torch.tensor(O_DIHEDRAL, device=device, dtype=dtype),
            )
            residues.append(torch.stack([n_atom, ca_atom, c_atom, o_atom]))
            if index == length - 1:
                break
            psi = torsion_angles[index, 1]
            omega = torsion_angles[index, 2]
            phi_next = torsion_angles[index + 1, 0]
            next_n = place_atom(
                n_atom, ca_atom, c_atom, BOND_C_N, ANGLE_CA_C_N, psi
            )
            next_ca = place_atom(
                ca_atom, c_atom, next_n, BOND_N_CA, ANGLE_C_N_CA, omega
            )
            next_c = place_atom(
                c_atom, next_n, next_ca, BOND_CA_C, ANGLE_N_CA_C, phi_next
            )
            n_atom, ca_atom, c_atom = next_n, next_ca, next_c
        coords = torch.stack(residues)
        ca_center = coords[:, 1].mean(dim=0, keepdim=True)
        return coords - ca_center[None, :, :]


class DistogramHead(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        projection_size: int = 256,
        pair_width: int = 256,
        bins: int = 32,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.residue_projection = nn.Linear(
            hidden_size, projection_size, bias=False
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(projection_size * 2, pair_width),
            nn.SiLU(),
            nn.Linear(pair_width, bins),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.residue_projection(self.norm(hidden.float()))
        left = projected[pair_i]
        right = projected[pair_j]
        features = torch.cat(
            [(left - right).abs(), left * right], dim=-1
        )
        return self.pair_mlp(features)


class GeometryModel(nn.Module):
    def __init__(
        self,
        language_model: PeftModel,
        hidden_size: int = 4096,
        distogram_bins: int = 32,
        coordinate_mode: str = DEFAULT_COORDINATE_MODE,
    ):
        super().__init__()
        self.language_model = language_model
        self.hidden_size = hidden_size
        if coordinate_mode == "free":
            self.coordinate_head = CoordinateHead(hidden_size=hidden_size)
        elif coordinate_mode == "internal":
            self.coordinate_head = InternalCoordinateHead(hidden_size=hidden_size)
        else:
            raise ValueError(f"Unknown coordinate_mode={coordinate_mode}")
        self.distogram_head = DistogramHead(
            hidden_size=hidden_size, bins=distogram_bins
        )
        self.distogram_bins = distogram_bins
        self.coordinate_mode = coordinate_mode

    def text_tower(self) -> nn.Module:
        return self.language_model.base_model.model.model

    def forward(
        self,
        input_ids: torch.Tensor,
        marker_positions: torch.Tensor,
        pair_i: torch.Tensor | None = None,
        pair_j: torch.Tensor | None = None,
        relation_input_ids: torch.Tensor | None = None,
        relation_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.text_tower()(
            input_ids=input_ids.unsqueeze(0),
            use_cache=False,
        )
        marker_hidden = outputs.last_hidden_state[
            0, marker_positions.long()
        ]
        coords = self.coordinate_head(marker_hidden)
        if pair_i is None or pair_j is None:
            pair_i, pair_j = torch.triu_indices(
                len(marker_positions),
                len(marker_positions),
                offset=1,
                device=input_ids.device,
            )
        distogram_logits = self.distogram_head(
            marker_hidden, pair_i, pair_j
        )
        result = {
            "coords": coords,
            "distogram_logits": distogram_logits,
            "pair_i": pair_i,
            "pair_j": pair_j,
        }
        if relation_input_ids is not None:
            if relation_labels is None:
                raise ValueError("relation_labels are required with relation_input_ids")
            relation_outputs = self.language_model(
                input_ids=relation_input_ids.unsqueeze(0),
                labels=relation_labels.unsqueeze(0),
                use_cache=False,
            )
            result["relation_loss"] = relation_outputs.loss
        return result

    def head_state_dict(self) -> dict[str, Any]:
        return {
            "coordinate_head": self.coordinate_head.state_dict(),
            "distogram_head": self.distogram_head.state_dict(),
            "config": {
                "hidden_size": self.hidden_size,
                "distogram_bins": self.distogram_bins,
                "coordinate_scale": COORDINATE_SCALE,
                "coordinate_mode": self.coordinate_mode,
            },
        }

    def load_head_state_dict(self, state: dict[str, Any]) -> None:
        self.coordinate_head.load_state_dict(state["coordinate_head"])
        self.distogram_head.load_state_dict(state["distogram_head"])

    def save_geometry_pretrained(self, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
        self.language_model.save_pretrained(
            output / "adapter", safe_serialization=True
        )
        torch.save(self.head_state_dict(), output / "geometry_heads.pt")
        (output / "geometry_config.json").write_text(
            json.dumps(
                {
                    "hidden_size": self.hidden_size,
                    "coordinate_scale": COORDINATE_SCALE,
                    "distogram_bins": self.distogram_bins,
                    "coordinate_mode": self.coordinate_mode,
                    "target_modules": ALL_ATTN_TARGET_MODULES,
                },
                indent=2,
            )
            + "\n"
        )


def module_match_summary(model: PeftModel) -> dict[str, Any]:
    counts = {name: 0 for name in ALL_ATTN_TARGET_MODULES}
    parameters = {name: 0 for name in ALL_ATTN_TARGET_MODULES}
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
            f"LoRA module audit failed: missing={missing}, "
            f"unmatched={unmatched[:8]}"
        )
    return {
        "module_counts": counts,
        "trainable_parameters_by_suffix": parameters,
        "lora_parameters": int(sum(parameters.values())),
    }
