"""Rotation conventions and SMPL-X root geometry shared by data and losses."""
import functools
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def axis_angle_to_matrix(axis_angle):
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(1e-6)
    axis = axis_angle / angle
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    skew = skew.reshape(*axis.shape[:-1], 3, 3)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(skew.shape)
    return eye + angle.sin()[..., None] * skew + (1 - angle.cos()[..., None]) * (skew @ skew)


def matrix_to_rot6d(matrix):
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), -1)


def rot6d_to_matrix(rot6d):
    first, second = rot6d[..., :3], rot6d[..., 3:]
    b1 = F.normalize(first, dim=-1)
    b2 = F.normalize(second - (b1 * second).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), -1)


def matrix_to_axis_angle(matrix):
    # Select a well-conditioned quaternion branch, including rotations near pi.
    flat = matrix.reshape(-1, 3, 3)
    r00, r11, r22 = flat[:, 0, 0], flat[:, 1, 1], flat[:, 2, 2]
    trace = r00 + r11 + r22
    positive = trace > 0
    case_x = (~positive) & (r00 >= r11) & (r00 >= r22)
    case_y = (~positive) & (~case_x) & (r11 >= r22)
    case_z = (~positive) & (~case_x) & (~case_y)
    quaternion = matrix.new_zeros((len(flat), 4))
    for mask, component in ((positive, 0), (case_x, 1), (case_y, 2), (case_z, 3)):
        r = flat[mask]
        if component == 0:
            s = (trace[mask] + 1).clamp_min(1e-6).sqrt() * 2
            values = (s / 4, (r[:, 2, 1] - r[:, 1, 2]) / s,
                      (r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 1, 0] - r[:, 0, 1]) / s)
        elif component == 1:
            s = (1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2]).clamp_min(1e-6).sqrt() * 2
            values = ((r[:, 2, 1] - r[:, 1, 2]) / s, s / 4,
                      (r[:, 0, 1] + r[:, 1, 0]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s)
        elif component == 2:
            s = (1 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2]).clamp_min(1e-6).sqrt() * 2
            values = ((r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 0, 1] + r[:, 1, 0]) / s,
                      s / 4, (r[:, 1, 2] + r[:, 2, 1]) / s)
        else:
            s = (1 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1]).clamp_min(1e-6).sqrt() * 2
            values = ((r[:, 1, 0] - r[:, 0, 1]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                      (r[:, 1, 2] + r[:, 2, 1]) / s, s / 4)
        quaternion[mask] = torch.stack(values, -1)
    quaternion = F.normalize(quaternion, dim=-1, eps=1e-6)
    quaternion = quaternion * torch.where(quaternion[:, :1] < 0, -1.0, 1.0)
    xyz = quaternion[:, 1:]
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sin_half, quaternion[:, :1].clamp_min(1e-6))
    scale = torch.where(sin_half > 1e-6, angle / sin_half.clamp_min(1e-6), 2.0)
    return (xyz * scale).reshape(*matrix.shape[:-2], 3)


def stabilize_axis_angle(axis_angle, angle_eps=1e-4):
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle.clamp_min(1e-6)
    wrapped = torch.remainder(angle + math.pi, 2 * math.pi) - math.pi
    return axis * wrapped.clamp(-math.pi + angle_eps, math.pi - angle_eps)


@functools.lru_cache(maxsize=4)
def _smplx_root_basis(models_dir):
    path = Path(models_dir) / "smplx" / "SMPLX_NEUTRAL.npz"
    with np.load(path, allow_pickle=True) as model:
        regressor = model["J_regressor"]
        if regressor.dtype == object:
            regressor = regressor.item()
        if hasattr(regressor, "toarray"):
            regressor = regressor.toarray()
        root_weights = np.asarray(regressor[0], dtype=np.float32)
        template = root_weights @ model["v_template"]
        basis = np.einsum("v,vci->ci", root_weights, model["shapedirs"][:, :, :10])
    return torch.tensor(template, dtype=torch.float32), torch.tensor(basis, dtype=torch.float32)


def smplx_rest_root(betas, models_dir):
    """SMPL pelvis is J(beta) + transl, independent of global orientation."""
    template, basis = _smplx_root_basis(str(models_dir))
    return template.to(betas) + betas @ basis.to(betas).T
