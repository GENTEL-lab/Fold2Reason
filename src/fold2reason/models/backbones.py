#!/usr/bin/env python3
"""Strict loaders for non-Qwen3.5 model-level replication checkpoints."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
)


def ensure_internvl_transformers_compatibility() -> None:
    """Bridge InternVL remote code to the Transformers 5 loading contract."""
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}  # type: ignore[attr-defined]


def local_model_type(model_path: Path) -> str:
    return str(json.loads((model_path / "config.json").read_text())["model_type"])


def load_model_level_tokenizer(model_path: Path) -> Any:
    model_type = local_model_type(model_path)
    if model_type == "internvl_chat":
        return AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=False,
            fix_mistral_regex=True,
        )
    return AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        fix_mistral_regex=model_type == "mistral3",
    )


def is_mllama_base_tokenizer(tokenizer: Any) -> bool:
    name = str(getattr(tokenizer, "name_or_path", ""))
    return "Llama-3.2-11B-Vision" in name and "Instruct" not in name


def render_mllama_base_completion(
    system: str,
    user: str,
    answer_prefix: str = "Answer:\n",
) -> str:
    """Meta's base model uses plain completion after the BOS token."""
    return f"{system.strip()}\n\n{user.strip()}\n\n{answer_prefix}"


def is_ministral3_base_tokenizer(tokenizer: Any) -> bool:
    name = str(getattr(tokenizer, "name_or_path", ""))
    return "Ministral-3-8B-Base-2512" in name


def render_ministral3_base_completion(
    system: str,
    user: str,
    answer_prefix: str = "Assistant:\n",
) -> str:
    """Frozen plain-completion protocol for the chat-template-free Base model."""
    return (
        f"System:\n{system.strip()}\n\n"
        f"User:\n{user.strip()}\n\n{answer_prefix}"
    )


def load_ministral3_text_model(
    model_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load only the 8.4B text tower from a full Ministral 3 checkpoint."""
    full_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if getattr(full_config, "model_type", None) != "mistral3":
        raise RuntimeError(f"Expected mistral3 config at {model_path}")
    text_config = full_config.text_config
    text_config._attn_implementation = "sdpa"
    text_config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=text_config,
        key_mapping={r"^language_model\.": ""},
        dtype=torch.bfloat16,
        device_map={"": device.index},
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    # The checkpoint ships generation_config.max_length=262144. Every
    # benchmark supplies an explicit max_new_tokens contract, so retaining
    # that default only triggers conflicting-limit warnings and can inflate
    # generation-cache planning.
    model.generation_config.max_length = None
    return model


def load_internvl35_text_model(
    model_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    ensure_internvl_transformers_compatibility()
    full_model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
    )
    if not hasattr(full_model, "language_model"):
        raise RuntimeError("InternVL checkpoint does not expose language_model")
    language_model = full_model.language_model
    full_model.language_model = None
    del full_model
    gc.collect()
    language_model = language_model.to(device)
    language_model.config.use_cache = False
    language_model.config._attn_implementation = "sdpa"
    language_model.eval()
    return language_model
