#!/usr/bin/env python3
"""Differentiable global and local backbone losses for 1k-v2."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


DISTOGRAM_BOUNDARIES = torch.tensor(
    [2.0]
    + [2.5 + 0.5 * index for index in range(28)]
    + [18.0, 20.0],
    dtype=torch.float32,
)


def safe_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    weights = weights.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(eps)


@torch.no_grad()
def stop_gradient_kabsch(
    predicted_ca: torch.Tensor,
    target_ca: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = weights.float()
    normalized = weights / weights.sum().clamp_min(1e-8)
    predicted_center = (
        predicted_ca.float() * normalized[:, None]
    ).sum(dim=0)
    target_center = (target_ca.float() * normalized[:, None]).sum(dim=0)
    predicted_centered = predicted_ca.float() - predicted_center
    target_centered = target_ca.float() - target_center
    covariance = (
        predicted_centered * normalized[:, None]
    ).T @ target_centered
    left, _, right_t = torch.linalg.svd(covariance)
    correction = torch.eye(
        3, dtype=torch.float32, device=predicted_ca.device
    )
    correction[-1, -1] = torch.sign(
        torch.det(left @ right_t)
    ).clamp(min=-1.0, max=1.0)
    rotation = left @ correction @ right_t
    return rotation, predicted_center, target_center


def align_coordinates(
    predicted: torch.Tensor,
    target: torch.Tensor,
    residue_weights: torch.Tensor,
) -> torch.Tensor:
    rotation, predicted_center, target_center = stop_gradient_kabsch(
        predicted[:, 1], target[:, 1], residue_weights
    )
    return (
        predicted.float() - predicted_center[None, None, :]
    ) @ rotation + target_center[None, None, :]


def normalize_vector(
    vector: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(
        vector, dim=-1, keepdim=True
    ).clamp_min(eps)


def dihedral_sincos(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
) -> torch.Tensor:
    b0 = a - b
    b1 = c - b
    b2 = d - c
    b1_unit = normalize_vector(b1)
    v = b0 - (b0 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    w = b2 - (b2 * b1_unit).sum(dim=-1, keepdim=True) * b1_unit
    v = normalize_vector(v)
    w = normalize_vector(w)
    cosine = (v * w).sum(dim=-1)
    sine = (torch.cross(b1_unit, v, dim=-1) * w).sum(dim=-1)
    return torch.stack([sine, cosine], dim=-1)


def predicted_torsions(coords: torch.Tensor) -> torch.Tensor:
    length = len(coords)
    torsions = torch.zeros(
        length,
        3,
        2,
        dtype=torch.float32,
        device=coords.device,
    )
    if length < 2:
        return torsions
    torsions[1:, 0] = dihedral_sincos(
        coords[:-1, 2],
        coords[1:, 0],
        coords[1:, 1],
        coords[1:, 2],
    )
    torsions[:-1, 1] = dihedral_sincos(
        coords[:-1, 0],
        coords[:-1, 1],
        coords[:-1, 2],
        coords[1:, 0],
    )
    torsions[:-1, 2] = dihedral_sincos(
        coords[:-1, 1],
        coords[:-1, 2],
        coords[1:, 0],
        coords[1:, 1],
    )
    return torsions


def vector_distance(
    left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    return torch.linalg.vector_norm(left.float() - right.float(), dim=-1)


def local_backbone_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_weights: torch.Tensor,
) -> torch.Tensor:
    losses = []
    weights = []
    for left_atom, right_atom in ((0, 1), (1, 2), (2, 3)):
        predicted_length = vector_distance(
            predicted[:, left_atom], predicted[:, right_atom]
        )
        target_length = vector_distance(
            target[:, left_atom], target[:, right_atom]
        )
        losses.append(
            F.smooth_l1_loss(
                predicted_length,
                target_length,
                reduction="none",
                beta=0.1,
            )
        )
        weights.append(residue_weights * residue_mask.float())
    if len(predicted) > 1:
        neighbor_mask = residue_mask[:-1] & residue_mask[1:]
        neighbor_weight = (
            residue_weights[:-1] * residue_weights[1:]
        ).sqrt() * neighbor_mask.float()
        for predicted_length, target_length in (
            (
                vector_distance(predicted[:-1, 2], predicted[1:, 0]),
                vector_distance(target[:-1, 2], target[1:, 0]),
            ),
            (
                vector_distance(predicted[:-1, 1], predicted[1:, 1]),
                vector_distance(target[:-1, 1], target[1:, 1]),
            ),
        ):
            losses.append(
                F.smooth_l1_loss(
                    predicted_length,
                    target_length,
                    reduction="none",
                    beta=0.1,
                )
            )
            weights.append(neighbor_weight)
    numerator = sum(
        (loss * weight).sum() for loss, weight in zip(losses, weights)
    )
    denominator = sum(weight.sum() for weight in weights)
    return numerator / denominator.clamp_min(1e-8)


def balanced_contact_loss(
    predicted_distance: torch.Tensor,
    target_contact: torch.Tensor,
) -> torch.Tensor:
    logits = (8.0 - predicted_distance) / 1.0
    positive = target_contact.bool()
    negative = ~positive
    terms = []
    if positive.any():
        terms.append(
            F.binary_cross_entropy_with_logits(
                logits[positive],
                torch.ones_like(logits[positive]),
            )
        )
    if negative.any():
        terms.append(
            F.binary_cross_entropy_with_logits(
                logits[negative],
                torch.zeros_like(logits[negative]),
            )
        )
    return torch.stack(terms).mean()


def geometry_loss(
    outputs: dict[str, torch.Tensor],
    row: dict[str, Any],
    device: torch.device,
    loss_weights: dict[str, float] | None = None,
    contrastive_mode: str = "both",
    contrastive_margin: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted = outputs["coords"].float()
    target = row["target_coords"].to(device=device, dtype=torch.float32)
    residue_mask = row["residue_mask"].to(device=device)
    residue_weights = row["residue_weights"].to(
        device=device, dtype=torch.float32
    )
    effective_weight = residue_weights * residue_mask.float()
    pair_i = outputs["pair_i"]
    pair_j = outputs["pair_j"]
    pair_valid = residue_mask[pair_i] & residue_mask[pair_j]
    separation = pair_j - pair_i

    aligned = align_coordinates(
        predicted, target, effective_weight.clamp_min(1e-4)
    )
    coordinate_error = F.smooth_l1_loss(
        aligned,
        target,
        reduction="none",
        beta=0.5,
    ).mean(dim=(-1, -2))
    coordinate_loss = safe_weighted_mean(
        coordinate_error, effective_weight
    )

    predicted_distance = vector_distance(
        predicted[pair_i, 1], predicted[pair_j, 1]
    )
    target_distance = vector_distance(
        target[pair_i, 1], target[pair_j, 1]
    )
    pair_weight = (
        effective_weight[pair_i] * effective_weight[pair_j]
    ).sqrt()
    pair_error = F.smooth_l1_loss(
        predicted_distance.clamp(max=20.0),
        target_distance.clamp(max=20.0),
        reduction="none",
        beta=0.5,
    )
    local_pair = pair_valid & (separation < 6)
    long_pair = pair_valid & (separation >= 6)
    local_pair_loss = safe_weighted_mean(
        pair_error[local_pair], pair_weight[local_pair]
    )
    long_pair_loss = safe_weighted_mean(
        pair_error[long_pair], pair_weight[long_pair]
    )
    pair_distance_loss = 0.3 * local_pair_loss + 0.7 * long_pair_loss

    target_contact = target_distance[long_pair] < 8.0
    contact_loss = balanced_contact_loss(
        predicted_distance[long_pair], target_contact
    )

    boundaries = DISTOGRAM_BOUNDARIES.to(device=device)
    distogram_target = torch.bucketize(target_distance, boundaries)
    distogram_error = F.cross_entropy(
        outputs["distogram_logits"],
        distogram_target,
        reduction="none",
    )
    distogram_loss = safe_weighted_mean(
        distogram_error[pair_valid], pair_weight[pair_valid]
    )

    # Input-specific invariant geometry objective.  The same distogram logits
    # must score the matched structure better than length/topology-controlled
    # wrong structures.  This routes the signal through residue hidden states
    # and the distogram head without making Cartesian orientation observable.
    contrastive_terms: dict[str, torch.Tensor] = {}
    for name, field in (
        ("hard", "hard_negative_coords"),
        ("structured", "structured_negative_coords"),
    ):
        if field not in row:
            contrastive_terms[name] = predicted_distance.new_zeros(())
            continue
        negative = row[field].to(device=device, dtype=torch.float32)
        negative_distance = vector_distance(
            negative[pair_i, 1], negative[pair_j, 1]
        )
        negative_target = torch.bucketize(negative_distance, boundaries)
        negative_error = F.cross_entropy(
            outputs["distogram_logits"], negative_target, reduction="none"
        )
        matched_ce = safe_weighted_mean(
            distogram_error[long_pair], pair_weight[long_pair]
        )
        negative_ce = safe_weighted_mean(
            negative_error[long_pair], pair_weight[long_pair]
        )
        contrastive_terms[name] = F.relu(
            predicted_distance.new_tensor(contrastive_margin)
            + matched_ce
            - negative_ce
        )
    selected = {
        "hard": ("hard",),
        "structured": ("structured",),
        "both": ("hard", "structured"),
    }[contrastive_mode]
    contrastive_loss = torch.stack(
        [contrastive_terms[name] for name in selected]
    ).mean()

    backbone_loss = local_backbone_loss(
        predicted, target, residue_mask, effective_weight
    )

    target_torsion = row["torsion_sincos"].to(
        device=device, dtype=torch.float32
    )
    torsion_mask = row["torsion_mask"].to(device=device)
    torsion_prediction = predicted_torsions(predicted)
    torsion_error = (
        torsion_prediction - target_torsion
    ).square().sum(dim=-1)
    torsion_weight = (
        effective_weight[:, None].expand_as(torsion_error)
        * torsion_mask.float()
    )
    torsion_loss = safe_weighted_mean(
        torsion_error, torsion_weight
    )

    valid_ca = predicted[residue_mask, 1]
    predicted_rg = torch.sqrt(
        (
            valid_ca - valid_ca.mean(dim=0, keepdim=True)
        ).square().sum(dim=-1).mean().clamp_min(1e-8)
    )
    target_rg = torch.tensor(
        float(row["radius_of_gyration"]),
        dtype=torch.float32,
        device=device,
    )
    rg_loss = F.smooth_l1_loss(
        torch.log(predicted_rg.clamp_min(1e-4)),
        torch.log(target_rg.clamp_min(1e-4)),
        beta=0.1,
    )

    components = {
        "aligned_coord": coordinate_loss,
        "pair_distance": pair_distance_loss,
        "contact": contact_loss,
        "distogram": distogram_loss,
        "backbone_local": backbone_loss,
        "torsion": torsion_loss,
        "radius_of_gyration": rg_loss,
        "contrastive_hard": contrastive_terms["hard"],
        "contrastive_structured": contrastive_terms["structured"],
        "input_contrastive": contrastive_loss,
        "predicted_rg": predicted_rg.detach(),
        "target_rg": target_rg.detach(),
        "positive_contacts": target_contact.sum().float().detach(),
        "valid_long_pairs": long_pair.sum().float().detach(),
    }
    weights = {
        "aligned_coord": 1.0,
        "pair_distance": 1.0,
        "contact": 0.5,
        "distogram": 0.3,
        "backbone_local": 0.2,
        "torsion": 0.2,
        "radius_of_gyration": 0.1,
        "input_contrastive": 0.0,
    }
    if loss_weights is not None:
        weights.update(loss_weights)
    total = (
        weights["aligned_coord"] * coordinate_loss
        + weights["pair_distance"] * pair_distance_loss
        + weights["contact"] * contact_loss
        + weights["distogram"] * distogram_loss
        + weights["backbone_local"] * backbone_loss
        + weights["torsion"] * torsion_loss
        + weights["radius_of_gyration"] * rg_loss
        + weights["input_contrastive"] * contrastive_loss
    )
    components["total"] = total
    return total, components
