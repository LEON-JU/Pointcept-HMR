from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import torch

_EXT = None
_EXT_LOAD_ERROR: Exception | None = None
_SOURCE_HASH: str | None = None

_PKG_DIR = Path(
    __file__).resolve().parents[5] / "libs/human_pose_ops/pointnet2"
_SOURCE_FILES = (
    _PKG_DIR / "pointnet2_ext.cpp",
    _PKG_DIR / "pointnet2_ext_cuda.cu",
)
_BUILD_HINT = "Build it once with `bash run_scripts/build_pointnet2_ext.sh` before training or eval."


def _load_extension():
    global _EXT
    global _EXT_LOAD_ERROR
    if _EXT is not None or _EXT_LOAD_ERROR is not None:
        return _EXT
    try:
        try:
            _EXT = importlib.import_module(f"{__name__}._C")
        except Exception:
            _EXT = importlib.import_module("pointnet2._C")
    except Exception as exc:  # pragma: no cover
        _EXT_LOAD_ERROR = exc
    return _EXT


def _source_hash() -> str:
    global _SOURCE_HASH
    if _SOURCE_HASH is not None:
        return _SOURCE_HASH
    digest = hashlib.sha256()
    for path in _SOURCE_FILES:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    _SOURCE_HASH = digest.hexdigest()[:16]
    return _SOURCE_HASH


def pointnet2_ext_available() -> bool:
    return _load_extension() is not None


def pointnet2_ext_build_info() -> dict[str, object]:
    ext = _load_extension()
    info: dict[str, object] = {
        "available": ext is not None,
        "module_name": f"{__name__}._C",
        "module_path":
        str(Path(ext.__file__).resolve()) if ext is not None else None,
        "load_error":
        repr(_EXT_LOAD_ERROR) if _EXT_LOAD_ERROR is not None else None,
        "source_hash": _source_hash(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["current_device"] = int(torch.cuda.current_device())
        info["device_capability"] = ".".join(
            str(v) for v in torch.cuda.get_device_capability())
    return info


def _require_extension():
    ext = _load_extension()
    if ext is None:
        detail = f" load_error={_EXT_LOAD_ERROR!r}" if _EXT_LOAD_ERROR is not None else ""
        raise RuntimeError("Precompiled pointnet2_ext is unavailable."
                           f"{detail} {_BUILD_HINT}")
    return ext


def _shape_str(tensor: torch.Tensor) -> str:
    return str(tuple(int(v) for v in tensor.shape))


def _offset_to_bounds(offset: torch.Tensor) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    for end_t in offset.tolist():
        end = int(end_t)
        bounds.append((start, end))
        start = end
    return bounds


def _require_cuda_float32_matrix(name: str,
                                 tensor: torch.Tensor,
                                 *,
                                 last_dim: int | None = None) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(tensor)!r}.")
    if tensor.device.type != "cuda":
        raise ValueError(
            f"{name} must be a CUDA tensor, got device={tensor.device}.")
    if tensor.dtype != torch.float32:
        raise TypeError(
            f"{name} must be float32 for pointnet2_ext, got dtype={tensor.dtype}."
        )
    if tensor.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D tensor, got shape={_shape_str(tensor)}.")
    if last_dim is not None and int(tensor.shape[1]) != int(last_dim):
        raise ValueError(
            f"{name} must have shape (N,{int(last_dim)}), got shape={_shape_str(tensor)}."
        )
    return tensor.contiguous()


def _require_index_tensor(name: str, tensor: torch.Tensor, *,
                          device: torch.device, ndim: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(tensor)!r}.")
    if tensor.device != device:
        raise ValueError(
            f"{name} must be on device={device}, got device={tensor.device}.")
    if tensor.ndim != int(ndim):
        raise ValueError(
            f"{name} must be a {int(ndim)}D tensor, got shape={_shape_str(tensor)}."
        )
    if tensor.dtype not in {torch.int32, torch.int64}:
        raise TypeError(
            f"{name} must have integer dtype, got dtype={tensor.dtype}.")
    return tensor.to(dtype=torch.long).contiguous()


def _validate_offset(
    name: str,
    offset: torch.Tensor,
    *,
    device: torch.device,
    total_size: int,
    allow_empty_segments: bool,
) -> torch.Tensor:
    offset = _require_index_tensor(name, offset, device=device, ndim=1)
    if int(offset.numel()) == 0:
        if total_size != 0:
            raise ValueError(
                f"{name} is empty but expected final value {total_size}.")
        return offset
    if int(offset[-1].item()) != int(total_size):
        raise ValueError(
            f"{name}[-1] must equal {total_size}, got {int(offset[-1].item())}."
        )
    prev = torch.cat([offset.new_zeros(1), offset[:-1]])
    counts = offset - prev
    if bool((counts < 0).any().item()):
        bad = torch.nonzero(counts < 0, as_tuple=False).reshape(-1).tolist()
        raise ValueError(
            f"{name} must be nondecreasing. bad_indices={bad[:8]} values={offset.tolist()[:10]}"
        )
    if not allow_empty_segments and bool((counts <= 0).any().item()):
        bad = torch.nonzero(counts <= 0, as_tuple=False).reshape(-1).tolist()
        raise ValueError(
            f"{name} contains empty segments, which pointnet2_ext does not allow. "
            f"bad_batch_indices={bad[:8]} values={offset.tolist()[:10]}")
    return offset


def _validate_idx(idx: torch.Tensor, *, n_points: int,
                  name: str) -> torch.Tensor:
    if int(idx.numel()) == 0:
        return idx
    min_idx = int(idx.min().item())
    max_idx = int(idx.max().item())
    if min_idx < 0 or max_idx >= int(n_points):
        raise ValueError(
            f"{name} must stay within [0, {int(n_points) - 1}], got min={min_idx} max={max_idx}."
        )
    return idx


def _normalize_radii(
    radii: float | torch.Tensor,
    *,
    new_offset: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(radii, torch.Tensor):
        if radii.device != device:
            radii = radii.to(device=device)
        if not radii.is_floating_point():
            raise TypeError(
                f"radii tensor must be floating point, got dtype={radii.dtype}."
            )
        radius_t = radii.to(dtype=torch.float32).reshape(-1)
    else:
        radius_t = torch.tensor([float(radii)],
                                device=device,
                                dtype=torch.float32)
    total = int(new_offset[-1].item()) if int(new_offset.numel()) > 0 else 0
    if int(radius_t.numel()) == 1:
        return radius_t.expand(total).contiguous()
    if int(radius_t.numel()) == int(new_offset.shape[0]):
        counts = torch.diff(torch.cat([new_offset.new_zeros(1), new_offset]))
        return torch.repeat_interleave(radius_t, counts, dim=0).contiguous()
    if int(radius_t.numel()) != total:
        raise ValueError(
            f"radii must have length 1, B, or total_queries={total}. Got {int(radius_t.numel())}."
        )
    return radius_t.contiguous()


def _reference_fps_single(coord: torch.Tensor, npoint: int) -> torch.Tensor:
    n = int(coord.shape[0])
    selected = torch.empty((npoint, ), device=coord.device, dtype=torch.long)
    if n <= 0:
        return selected.zero_()
    dist = torch.full((n, ),
                      float("inf"),
                      device=coord.device,
                      dtype=coord.dtype)
    farthest = 0
    for i in range(int(npoint)):
        selected[i] = int(farthest)
        center = coord[farthest:farthest + 1]
        dist = torch.minimum(dist, torch.sum((coord - center)**2, dim=-1))
        farthest = int(torch.argmax(dist).item())
    return selected


def _reference_ball_query_packed(
    xyz: torch.Tensor,
    offset: torch.Tensor,
    new_xyz: torch.Tensor,
    new_offset: torch.Tensor,
    radii: torch.Tensor,
    nsample: int,
) -> torch.Tensor:
    idx = torch.empty((int(new_xyz.shape[0]), int(nsample)),
                      device=xyz.device,
                      dtype=torch.long)
    q_start = 0
    for (p_start, p_end), q_end_t in zip(_offset_to_bounds(offset),
                                         new_offset.tolist(),
                                         strict=False):
        q_end = int(q_end_t)
        xyz_b = xyz[p_start:p_end]
        radius_b = radii[q_start:q_end]
        for local_q, center in enumerate(new_xyz[q_start:q_end]):
            dist2 = torch.sum((xyz_b - center.unsqueeze(0))**2, dim=-1)
            inside = torch.nonzero(dist2 <= float(radius_b[local_q].item())**2,
                                   as_tuple=False).reshape(-1)
            if int(inside.numel()) <= 0:
                idx[q_start + local_q] = int(p_start)
                continue
            base = inside[:1].expand(int(nsample)).clone()
            take = inside[:int(nsample)]
            base[:int(take.numel())] = take
            idx[q_start + local_q] = base.to(dtype=torch.long) + int(p_start)
        q_start = q_end
    return idx


def furthest_point_sample_packed(
        xyz: torch.Tensor, offset: torch.Tensor,
        npoint: int) -> tuple[torch.Tensor, torch.Tensor]:
    if int(npoint) <= 0:
        raise ValueError(f"npoint must be positive, got {npoint}.")
    _require_extension()
    xyz = _require_cuda_float32_matrix("xyz", xyz, last_dim=3)
    offset = _validate_offset("offset",
                              offset,
                              device=xyz.device,
                              total_size=int(xyz.shape[0]),
                              allow_empty_segments=False)
    if int(offset.numel()) == 0:
        return offset.new_zeros((0, ), dtype=torch.long), offset
    ext = _require_extension()
    return ext.furthest_point_sample_packed(xyz, offset, int(npoint))


class _GatherPointsFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(idx)
        ctx.n_points = int(points.shape[0])
        ext = _require_extension()
        return ext.gather_points_forward(points.contiguous(), idx.contiguous())

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (idx, ) = ctx.saved_tensors
        ext = _require_extension()
        grad_points = ext.gather_points_backward(grad_out.contiguous(),
                                                 idx.contiguous(),
                                                 int(ctx.n_points))
        return grad_points, None


def gather_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    _require_extension()
    points = _require_cuda_float32_matrix("points", points)
    idx = _require_index_tensor("idx", idx, device=points.device,
                                ndim=1).reshape(-1)
    idx = _validate_idx(idx, n_points=int(points.shape[0]), name="idx")
    return _GatherPointsFn.apply(points, idx)


def ball_query_packed(
    xyz: torch.Tensor,
    offset: torch.Tensor,
    new_xyz: torch.Tensor,
    new_offset: torch.Tensor,
    radii: float | torch.Tensor,
    nsample: int,
) -> torch.Tensor:
    if int(nsample) <= 0:
        raise ValueError(f"nsample must be positive, got {nsample}.")
    _require_extension()
    xyz = _require_cuda_float32_matrix("xyz", xyz, last_dim=3)
    new_xyz = _require_cuda_float32_matrix("new_xyz", new_xyz, last_dim=3)
    offset = _validate_offset("offset",
                              offset,
                              device=xyz.device,
                              total_size=int(xyz.shape[0]),
                              allow_empty_segments=False)
    new_offset = _validate_offset(
        "new_offset",
        new_offset,
        device=xyz.device,
        total_size=int(new_xyz.shape[0]),
        allow_empty_segments=False,
    )
    if int(offset.shape[0]) != int(new_offset.shape[0]):
        raise ValueError(
            f"offset and new_offset must have the same batch size, got {int(offset.shape[0])} vs {int(new_offset.shape[0])}."
        )
    radius_t = _normalize_radii(radii,
                                new_offset=new_offset,
                                device=xyz.device)
    if int(new_xyz.shape[0]) == 0:
        return offset.new_zeros((0, int(nsample)), dtype=torch.long)
    ext = _require_extension()
    return ext.ball_query_packed(xyz, offset, new_xyz, new_offset, radius_t,
                                 int(nsample))


class _GroupPointsFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(idx)
        ctx.n_points = int(points.shape[0])
        ext = _require_extension()
        return ext.group_points_forward(points.contiguous(), idx.contiguous())

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (idx, ) = ctx.saved_tensors
        ext = _require_extension()
        grad_points = ext.group_points_backward(grad_out.contiguous(),
                                                idx.contiguous(),
                                                int(ctx.n_points))
        return grad_points, None


def group_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    _require_extension()
    points = _require_cuda_float32_matrix("points", points)
    idx = _require_index_tensor("idx", idx, device=points.device, ndim=2)
    idx = _validate_idx(idx, n_points=int(points.shape[0]), name="idx")
    return _GroupPointsFn.apply(points, idx)


__all__ = [
    "ball_query_packed",
    "furthest_point_sample_packed",
    "gather_points",
    "group_points",
    "pointnet2_ext_available",
    "pointnet2_ext_build_info",
]
