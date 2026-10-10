from __future__ import annotations

import torch
import torch.nn as nn

from .ops.pointnet2 import ball_query_packed, furthest_point_sample_packed, gather_points, group_points


def _reference_fps(coord, offset, npoint):
    indices, out_offset, start = [], [], 0
    for end_t in offset.tolist():
        end = int(end_t)
        points = coord[start:end]
        if points.shape[0] == 0:
            raise ValueError("PointNet++ requires non-empty samples")
        chosen = torch.zeros(int(npoint),
                             dtype=torch.long,
                             device=coord.device)
        distance = torch.full((points.shape[0], ),
                              float("inf"),
                              device=coord.device)
        last = 0
        for i in range(int(npoint)):
            chosen[i] = last
            distance = torch.minimum(distance,
                                     ((points - points[last])**2).sum(-1))
            last = int(distance.argmax())
        indices.append(chosen + start)
        start = end
        out_offset.append(start)
    return torch.cat(indices), torch.tensor(out_offset,
                                            dtype=torch.long,
                                            device=coord.device)


def _fps(coord, offset, npoint, backend):
    if backend == "cuda":
        return furthest_point_sample_packed(coord.float().contiguous(),
                                            offset.long().contiguous(),
                                            int(npoint))
    return _reference_fps(coord, offset, npoint)


def _ball_query(coord, offset, centers, center_offset, radius, nsample,
                backend):
    if backend == "cuda":
        return ball_query_packed(coord.float().contiguous(),
                                 offset.long().contiguous(),
                                 centers.float().contiguous(),
                                 center_offset.long().contiguous(), radius,
                                 int(nsample))
    result, pstart, qstart = [], 0, 0
    for pend_t, qend_t in zip(offset.tolist(), center_offset.tolist()):
        pend, qend = int(pend_t), int(qend_t)
        cloud = coord[pstart:pend]
        radius_values = radius[qstart:qend] if isinstance(
            radius, torch.Tensor) else None
        for query_i, center in enumerate(centers[qstart:qend]):
            radius_i = float(radius_values[query_i]
                             ) if radius_values is not None else float(radius)
            d2 = ((cloud - center)**2).sum(-1)
            inside = torch.where(d2 <= radius_i**2)[0]
            if not inside.numel(): inside = d2.argmin().reshape(1)
            selected = inside[:nsample]
            if selected.numel() < nsample:
                selected = torch.cat([
                    selected, selected[:1].expand(nsample - selected.numel())
                ])
            result.append(selected + pstart)
        pstart, qstart = pend, qend
    return torch.stack(result)


def _gather(x, idx, backend):
    return gather_points(x.float().contiguous(),
                         idx.reshape(-1).long()).reshape(
                             *idx.shape,
                             x.shape[-1]) if backend == "cuda" else x[idx]


def _group(x, idx, backend):
    if backend == "cuda":
        return group_points(x.float().contiguous(),
                            idx.reshape(idx.shape[0], -1).long())
    return x[idx]


def _mlp(in_channels, widths):
    layers = []
    current = in_channels
    for width in widths:
        layers.extend((nn.Conv2d(current, width, 1, bias=False),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True)))
        current = width
    return nn.Sequential(*layers)


class PointNetPPGlobalFusion(nn.Module):
    """MSG PointNet++ global feature used by the migrated regression head."""

    def __init__(self,
                 in_channels,
                 backend="cuda",
                 max_centers=64,
                 radii=(0.08, 0.16, 0.32),
                 nsamples=(16, 32, 64)):
        super().__init__()
        self.backend = backend
        self.max_centers = int(max_centers)
        self.radii = tuple(radii)
        self.nsamples = tuple(nsamples)
        self.branches = nn.ModuleList([
            _mlp(int(in_channels) + 3, widths)
            for widths in ((64, 64, 128), (64, 96, 128), (64, 96, 128))
        ])
        self.global_mlp = _mlp(3 + 384, (256, 512, 1024))

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, coord, feat, offset):
        coord, feat, offset = coord.float().contiguous(), feat.float(
        ).contiguous(), offset.long().contiguous()
        center_idx, center_offset = _fps(coord, offset, self.max_centers,
                                         self.backend)
        centers = _gather(coord, center_idx, self.backend).reshape(-1, 3)
        counts = torch.diff(torch.cat([offset.new_zeros(1), offset]))
        scales = []
        start = 0
        for end_t in offset.tolist():
            end = int(end_t)
            scales.append(
                torch.linalg.vector_norm(coord[start:end].amax(0) -
                                         coord[start:end].amin(0)).clamp_min(
                                             1e-3))
            start = end
        scales = torch.stack(scales)
        branches = []
        for radius, nsample, mlp in zip(self.radii, self.nsamples,
                                        self.branches):
            scaled_radius = torch.repeat_interleave(scales * float(radius),
                                                    self.max_centers)
            idx = _ball_query(coord, offset, centers, center_offset,
                              scaled_radius, nsample, self.backend)
            grouped_xyz = _group(coord, idx,
                                 self.backend) - centers.unsqueeze(1)
            grouped_feat = _group(feat, idx, self.backend)
            grouped = torch.cat([grouped_xyz, grouped_feat],
                                -1).permute(0, 2,
                                            1).unsqueeze(-1).contiguous()
            branches.append(mlp(grouped).amax(dim=2).squeeze(-1))
        msg = torch.cat(branches, dim=1)
        center_counts = torch.diff(
            torch.cat([center_offset.new_zeros(1), center_offset]))
        if not torch.all(center_counts == center_counts[0]):
            raise RuntimeError(
                "PointNet++ global fusion expects fixed center count")
        dense_coord = centers.view(len(offset), int(center_counts[0]), 3)
        dense_msg = msg.view(len(offset), int(center_counts[0]), -1)
        global_input = torch.cat([dense_coord, dense_msg],
                                 -1).permute(0, 2,
                                             1).unsqueeze(-1).contiguous()
        return self.global_mlp(global_input).amax(dim=2).squeeze(-1)
