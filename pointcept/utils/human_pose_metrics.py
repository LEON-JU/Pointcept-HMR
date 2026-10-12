"""Metrics used by the standalone human-pose evaluators."""

from __future__ import annotations

import numpy as np
import torch


M_TO_MM = 1000.0


def _compute_jpe(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return torch.linalg.vector_norm(pred - target, dim=-1).mean(dim=-1).numpy()


def _align_by_pelvis(
    pred_joints: torch.Tensor,
    target_joints: torch.Tensor,
    pelvis_idxs: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_pelvis = pred_joints[:, list(pelvis_idxs)].mean(dim=1, keepdim=True)
    target_pelvis = target_joints[:, list(pelvis_idxs)].mean(dim=1, keepdim=True)
    return pred_joints - pred_pelvis, target_joints - target_pelvis


def _similarity_transform(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Apply the optimal similarity transform from ``pred`` to ``target``."""

    pred = pred.permute(0, 2, 1)
    target = target.permute(0, 2, 1)
    mu_pred = pred.mean(dim=-1, keepdim=True)
    mu_target = target.mean(dim=-1, keepdim=True)
    centered_pred = pred - mu_pred
    centered_target = target - mu_target
    variance = centered_pred.square().sum(dim=1).sum(dim=1)
    covariance = centered_pred.bmm(centered_target.transpose(1, 2))

    u, _, vh = torch.linalg.svd(covariance)
    v = vh.transpose(1, 2)
    correction = torch.eye(3, device=pred.device, dtype=pred.dtype).expand(
        pred.shape[0], -1, -1
    ).clone()
    correction[:, -1, -1] = torch.sign(
        torch.linalg.det(u.bmm(v.transpose(1, 2)))
    )
    rotation = v.bmm(correction.bmm(u.transpose(1, 2)))
    scale = (rotation.bmm(covariance).diagonal(dim1=1, dim2=2).sum(dim=1) /
             variance.clamp_min(torch.finfo(pred.dtype).eps))
    translation = mu_target - scale[:, None, None] * rotation.bmm(mu_pred)
    return (scale[:, None, None] * rotation.bmm(pred) + translation).transpose(1, 2)


@torch.no_grad()
def compute_mpjpe_pa_mpjpe_mm(
    pred_joints: torch.Tensor,
    target_joints: torch.Tensor,
    pelvis_idxs: tuple[int, int] = (1, 2),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pelvis-aligned MPJPE and PA-MPJPE in millimeters."""

    pred = pred_joints.detach().to(device="cpu", dtype=torch.float32)
    target = target_joints.detach().to(device="cpu", dtype=torch.float32)
    pred, target = _align_by_pelvis(pred, target, pelvis_idxs)
    pred_pa = _similarity_transform(pred, target)
    mpjpe = torch.from_numpy(_compute_jpe(pred, target) * M_TO_MM)
    pa_mpjpe = torch.from_numpy(_compute_jpe(pred_pa, target) * M_TO_MM)
    return mpjpe, pa_mpjpe


@torch.no_grad()
def compute_pve_mm(
    pred_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
) -> torch.Tensor:
    """Return direct corresponding-vertex error in millimeters."""

    pred = pred_vertices.detach().to(dtype=torch.float32)
    target = target_vertices.detach().to(dtype=torch.float32)
    if pred.shape != target.shape:
        raise ValueError(
            "pred_vertices and target_vertices must have identical shapes, "
            f"got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError(f"Expected vertices shaped (B,V,3), got {tuple(pred.shape)}")
    return torch.linalg.vector_norm(pred - target, dim=-1).mean(dim=-1) * M_TO_MM


__all__ = ["compute_mpjpe_pa_mpjpe_mm", "compute_pve_mm"]
