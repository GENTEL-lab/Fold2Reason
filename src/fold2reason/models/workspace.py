#!/usr/bin/env python3
"""Shared entity/pair workspace consumed by frozen geometry and native LM readouts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from fold2reason.models.geometry import CoordinateHead, DistogramHead


def consume_geometry_head_initialization_rng(hidden_size: int) -> None:
    """Advance CPU RNG exactly like the two frozen heads, without creating them.

    The workspace follows the heads in the historical M3 constructor.  A literal
    no-head arm must therefore consume the same initialization draws to retain
    bit-identical trainable LoRA/workspace initialization across arms.
    """

    def linear(out_features: int, in_features: int, bias: bool = True) -> None:
        weight = torch.empty((out_features, in_features))
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias:
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(torch.empty(out_features), -bound, bound)

    linear(512, hidden_size)
    linear(12, 512)
    linear(256, hidden_size, bias=False)
    linear(256, 512)
    linear(32, 256)


class ResidualTransition(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.SiLU(),
            nn.Linear(width * 2, width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.mlp(self.norm(value))


class SpatialWorkspace(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        width: int = 256,
        memory_tokens: int = 16,
        fingerprint_dim: int = 64,
        max_sparse_pairs: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.width = width
        self.memory_tokens = memory_tokens
        self.fingerprint_dim = fingerprint_dim
        self.max_sparse_pairs = max_sparse_pairs
        self.input_norm = nn.LayerNorm(hidden_size)
        self.entity_down = nn.Linear(hidden_size, width)
        self.pair_mlp = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.message_transition = ResidualTransition(width)
        self.entity_transition = ResidualTransition(width)
        self.entity_up = nn.Linear(width, hidden_size, bias=False)
        # Near-identity start while retaining a non-zero gradient path from the
        # frozen decoder into the shared reduced entity/pair trunk at step 0.
        nn.init.normal_(self.entity_up.weight, std=1e-4)
        self.queries = nn.Parameter(torch.randn(memory_tokens, width) / math.sqrt(width))
        self.memory_up = nn.Linear(width, hidden_size, bias=False)
        nn.init.normal_(self.memory_up.weight, std=0.002)
        self.retrieval_projection = nn.Linear(width, fingerprint_dim, bias=False)

    def sparse_pairs(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        left_parts = []
        right_parts = []
        for offset in range(1, min(4, length - 1) + 1):
            left_parts.append(torch.arange(0, length - offset, device=device))
            right_parts.append(torch.arange(offset, length, device=device))
        local_i = torch.cat(left_parts) if left_parts else torch.empty(0, dtype=torch.long, device=device)
        local_j = torch.cat(right_parts) if right_parts else torch.empty(0, dtype=torch.long, device=device)
        remaining = max(self.max_sparse_pairs - len(local_i), 0)
        if remaining:
            long_i, long_j = torch.triu_indices(length, length, offset=5, device=device)
            if len(long_i) > remaining:
                selected = torch.linspace(0, len(long_i) - 1, remaining, device=device).long()
                long_i, long_j = long_i[selected], long_j[selected]
            pair_i = torch.cat([local_i, long_i])
            pair_j = torch.cat([local_j, long_j])
        else:
            pair_i, pair_j = local_i[: self.max_sparse_pairs], local_j[: self.max_sparse_pairs]
        return pair_i.long(), pair_j.long()

    def forward(self, marker_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        reduced = self.entity_down(self.input_norm(marker_hidden.float()))
        pair_i, pair_j = self.sparse_pairs(len(reduced), reduced.device)
        if len(pair_i):
            pair_features = torch.cat(
                [(reduced[pair_i] - reduced[pair_j]).abs(), reduced[pair_i] * reduced[pair_j]],
                dim=-1,
            )
            messages = self.pair_mlp(pair_features)
            aggregate = torch.zeros_like(reduced)
            counts = torch.zeros(len(reduced), 1, device=reduced.device, dtype=reduced.dtype)
            aggregate.index_add_(0, pair_i, messages)
            aggregate.index_add_(0, pair_j, messages)
            ones = torch.ones(len(pair_i), 1, device=reduced.device, dtype=reduced.dtype)
            counts.index_add_(0, pair_i, ones)
            counts.index_add_(0, pair_j, ones)
            reduced = self.message_transition(reduced + aggregate / counts.clamp_min(1.0))
        reduced = self.entity_transition(reduced)
        entity_memory = marker_hidden.float() + self.entity_up(reduced)
        attention = torch.softmax(
            self.queries @ reduced.transpose(0, 1) / math.sqrt(self.width),
            dim=-1,
        )
        pooled_reduced = attention @ reduced
        memory = self.memory_up(pooled_reduced)
        retrieval = F.normalize(self.retrieval_projection(pooled_reduced.mean(0)), dim=-1)
        return {
            "entity_memory": entity_memory,
            "memory_tokens": memory,
            "retrieval_embedding": retrieval,
            "workspace_pair_i": pair_i,
            "workspace_pair_j": pair_j,
            "memory_attention": attention,
        }


class WorkspaceGeometryModel(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        frozen_decoder_checkpoint: Path,
        workspace_width: int = 256,
        memory_tokens: int = 16,
        fingerprint_dim: int = 64,
        max_sparse_pairs: int = 2048,
        relation_uses_memory: bool = False,
        retrieval_temperature: float = 0.07,
        geometry_head_enabled: bool = True,
    ):
        super().__init__()
        self.language_model = language_model
        model_hidden_size = int(language_model.config.hidden_size)
        self.geometry_head_enabled = bool(geometry_head_enabled)
        if self.geometry_head_enabled:
            state = torch.load(
                frozen_decoder_checkpoint / "geometry_heads.pt",
                map_location="cpu",
                weights_only=False,
            )
            hidden_size = int(
                state.get("config", {}).get(
                    "hidden_size",
                    state["coordinate_head"]["norm.weight"].numel(),
                )
            )
            if hidden_size != model_hidden_size:
                raise RuntimeError(
                    "Frozen decoder/model hidden-size mismatch: "
                    f"decoder={hidden_size} model={model_hidden_size}"
                )
            self.coordinate_head = CoordinateHead(hidden_size=hidden_size)
            self.distogram_head = DistogramHead(hidden_size=hidden_size)
            self.coordinate_head.load_state_dict(state["coordinate_head"])
            self.distogram_head.load_state_dict(state["distogram_head"])
            for parameter in self.coordinate_head.parameters():
                parameter.requires_grad_(False)
            for parameter in self.distogram_head.parameters():
                parameter.requires_grad_(False)
        else:
            hidden_size = model_hidden_size
            consume_geometry_head_initialization_rng(hidden_size)
            self.coordinate_head = None
            self.distogram_head = None
        self.hidden_size = hidden_size
        self.workspace = SpatialWorkspace(
            hidden_size=hidden_size,
            width=workspace_width,
            memory_tokens=memory_tokens,
            fingerprint_dim=fingerprint_dim,
            max_sparse_pairs=max_sparse_pairs,
        )
        self.relation_uses_memory = relation_uses_memory
        self.retrieval_temperature = retrieval_temperature
        self.frozen_decoder_checkpoint = str(frozen_decoder_checkpoint)
        self.geometry_forward_calls = 0
        self.relation_forward_calls = 0

    def text_tower(self) -> nn.Module:
        return self.language_model.base_model.model.model

    def relation_forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        memory_tokens: torch.Tensor,
    ) -> torch.Tensor:
        self.relation_forward_calls += 1
        if not self.relation_uses_memory:
            return self.language_model(
                input_ids=input_ids.unsqueeze(0),
                labels=labels.unsqueeze(0),
                use_cache=False,
            ).loss
        prompt_embeddings = self.language_model.get_input_embeddings()(input_ids.unsqueeze(0))
        prefix = memory_tokens.to(dtype=prompt_embeddings.dtype).unsqueeze(0)
        inputs_embeds = torch.cat([prefix, prompt_embeddings], dim=1)
        prefix_labels = torch.full(
            (1, len(memory_tokens)),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        combined_labels = torch.cat([prefix_labels, labels.unsqueeze(0)], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
        )
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=combined_labels,
            use_cache=False,
        ).loss

    def _process_answer_logits(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        memory_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute only supervised answer-token logits.

        Process prompts are long while their answers contain only a handful of
        option tokens.  Calling the causal-LM wrapper would materialize
        ``sequence_length x vocabulary_size`` logits.  Selecting the causal
        hidden states before the LM head keeps M3-GT's teacher/student passes
        practical without changing the supervised probabilities.
        """
        target_positions = torch.nonzero(labels[1:] != -100, as_tuple=False).flatten() + 1
        if not len(target_positions):
            raise ValueError("process prompt contains no supervised answer tokens")
        prediction_positions = target_positions - 1
        if memory_tokens is None:
            outputs = self.text_tower()(input_ids=input_ids.unsqueeze(0), use_cache=False)
            hidden = outputs.last_hidden_state[0, prediction_positions]
        else:
            prompt_embeddings = self.language_model.get_input_embeddings()(
                input_ids.unsqueeze(0)
            )
            prefix = memory_tokens.to(dtype=prompt_embeddings.dtype).unsqueeze(0)
            inputs_embeds = torch.cat([prefix, prompt_embeddings], dim=1)
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device
            )
            outputs = self.text_tower()(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            hidden = outputs.last_hidden_state[
                0, prediction_positions + len(memory_tokens)
            ]
        logits = self.language_model.get_output_embeddings()(hidden)
        return logits.float(), labels[target_positions]

    @staticmethod
    def _other_rank_memory(memory_tokens: torch.Tensor) -> torch.Tensor:
        """Return a deterministic other-protein memory for the causal control."""
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return memory_tokens.detach().flip(0)
        gathered = [torch.empty_like(memory_tokens) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, memory_tokens.detach())
        return gathered[(dist.get_rank() + 1) % dist.get_world_size()]

    @staticmethod
    def _drop_memory_tokens(
        memory_tokens: torch.Tensor, probability: float
    ) -> torch.Tensor:
        if probability <= 0.0:
            return memory_tokens
        if probability >= 1.0:
            return torch.zeros_like(memory_tokens)
        keep = (
            torch.rand(
                (len(memory_tokens), 1),
                device=memory_tokens.device,
                dtype=torch.float32,
            )
            >= probability
        )
        return memory_tokens * keep.to(memory_tokens.dtype) / (1.0 - probability)

    def process_distillation_forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        matched_memory_tokens: torch.Tensor,
        *,
        student_ce: bool,
        distill: bool,
        teacher_mode: str = "matched",
        memory_dropout_probability: float = 0.0,
        distill_temperature: float = 2.0,
        base_kl: bool = False,
        specificity_margin: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """M3-GT process supervision and workspace-to-LM distillation.

        The student path never receives workspace tokens.  Consequently the
        saved LoRA adapter is sufficient at general-benchmark inference time.
        Teacher logits are detached for KL; matched teacher CE still trains the
        workspace and adapter to make the teacher informative.
        """
        other_memory = self._other_rank_memory(matched_memory_tokens)
        teacher_memory = (
            matched_memory_tokens if teacher_mode == "matched" else other_memory
        )
        teacher_memory = self._drop_memory_tokens(
            teacher_memory, memory_dropout_probability
        )
        teacher_logits, targets = self._process_answer_logits(
            input_ids, labels, teacher_memory
        )
        result: dict[str, torch.Tensor] = {
            "teacher_process_ce": F.cross_entropy(teacher_logits, targets),
            "process_answer_tokens": torch.tensor(
                len(targets), device=targets.device, dtype=torch.float32
            ),
        }

        need_student = student_ce or distill or base_kl
        student_logits = None
        if need_student:
            student_logits, student_targets = self._process_answer_logits(
                input_ids, labels, None
            )
            if not torch.equal(targets, student_targets):
                raise RuntimeError("teacher/student process targets are misaligned")
        if student_ce:
            result["student_process_ce"] = F.cross_entropy(student_logits, targets)
        if distill:
            temperature = float(distill_temperature)
            result["distill_kl"] = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits.detach() / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)
        if base_kl:
            with torch.no_grad(), self.language_model.disable_adapter():
                base_logits, base_targets = self._process_answer_logits(
                    input_ids, labels, None
                )
            if not torch.equal(targets, base_targets):
                raise RuntimeError("base/student process targets are misaligned")
            result["base_kl"] = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(base_logits, dim=-1),
                reduction="batchmean",
            )
        if specificity_margin is not None and teacher_mode == "matched":
            shuffled_logits, shuffled_targets = self._process_answer_logits(
                input_ids, labels, other_memory
            )
            if not torch.equal(targets, shuffled_targets):
                raise RuntimeError("matched/shuffled process targets are misaligned")
            shuffled_ce = F.cross_entropy(shuffled_logits, targets)
            result["shuffled_teacher_ce"] = shuffled_ce
            result["specificity_margin"] = F.relu(
                result["teacher_process_ce"]
                - shuffled_ce.detach()
                + float(specificity_margin)
            )
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        marker_positions: torch.Tensor,
        relation_input_ids: torch.Tensor | None = None,
        relation_labels: torch.Tensor | None = None,
        candidate_fingerprints: torch.Tensor | None = None,
        process_config: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.text_tower()(input_ids=input_ids.unsqueeze(0), use_cache=False)
        marker_hidden = outputs.last_hidden_state[0, marker_positions.long()]
        workspace = self.workspace(marker_hidden)
        entity_memory = workspace["entity_memory"]
        pair_i, pair_j = torch.triu_indices(
            len(entity_memory), len(entity_memory), offset=1, device=entity_memory.device
        )
        result = {**workspace, "pair_i": pair_i, "pair_j": pair_j}
        if self.geometry_head_enabled:
            self.geometry_forward_calls += 1
            result.update(
                {
                    "coords": self.coordinate_head(entity_memory).float(),
                    "distogram_logits": self.distogram_head(
                        entity_memory, pair_i, pair_j
                    ).float(),
                }
            )
        if candidate_fingerprints is not None:
            candidates = F.normalize(candidate_fingerprints.float(), dim=-1)
            logits = workspace["retrieval_embedding"].float() @ candidates.transpose(0, 1)
            logits = logits / self.retrieval_temperature
            result["retrieval_logits"] = logits
            result["retrieval_loss"] = F.cross_entropy(
                logits.unsqueeze(0),
                torch.zeros(1, dtype=torch.long, device=logits.device),
            )
        if relation_input_ids is not None:
            if relation_labels is None:
                raise ValueError("relation_labels required with relation_input_ids")
            result["relation_loss"] = self.relation_forward(
                relation_input_ids,
                relation_labels,
                workspace["memory_tokens"],
            )
        if process_config is not None:
            result.update(
                self.process_distillation_forward(
                    process_config["input_ids"],
                    process_config["labels"],
                    workspace["memory_tokens"],
                    student_ce=bool(process_config.get("student_ce", False)),
                    distill=bool(process_config.get("distill", False)),
                    teacher_mode=str(process_config.get("teacher_mode", "matched")),
                    memory_dropout_probability=float(
                        process_config.get("memory_dropout_probability", 0.0)
                    ),
                    distill_temperature=float(
                        process_config.get("distill_temperature", 2.0)
                    ),
                    base_kl=bool(process_config.get("base_kl", False)),
                    specificity_margin=process_config.get("specificity_margin"),
                )
            )
        return result

    def workspace_state_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace.state_dict(),
            "config": {
                "workspace_width": self.workspace.width,
                "hidden_size": self.hidden_size,
                "memory_tokens": self.workspace.memory_tokens,
                "fingerprint_dim": self.workspace.fingerprint_dim,
                "max_sparse_pairs": self.workspace.max_sparse_pairs,
                "relation_uses_memory": self.relation_uses_memory,
                "retrieval_temperature": self.retrieval_temperature,
                "frozen_decoder_checkpoint": self.frozen_decoder_checkpoint,
                "geometry_head_enabled": self.geometry_head_enabled,
            },
        }

    def save_pretrained(self, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
        self.language_model.save_pretrained(output / "adapter", safe_serialization=True)
        torch.save(self.workspace_state_dict(), output / "workspace.pt")
        if self.geometry_head_enabled:
            torch.save(
                {
                    "coordinate_head": self.coordinate_head.state_dict(),
                    "distogram_head": self.distogram_head.state_dict(),
                },
                output / "frozen_geometry_heads.pt",
            )
        (output / "workspace_config.json").write_text(
            json.dumps(self.workspace_state_dict()["config"], indent=2, sort_keys=True) + "\n"
        )

    def load_workspace_state_dict(self, state: dict[str, Any]) -> None:
        self.workspace.load_state_dict(state["workspace"])
