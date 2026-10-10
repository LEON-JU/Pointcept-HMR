from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path

import torch
import torch.nn as nn

_EXT = None
_EXT_LOAD_ERROR: Exception | None = None
_SOURCE_HASH: str | None = None

_PKG_DIR = Path(
    __file__).resolve().parents[5] / "libs/human_pose_ops/point3d_rope"
_SOURCE_FILES = (
    _PKG_DIR / "bindings.cpp",
    _PKG_DIR / "utonia_pointrope_cuda.cu",
)
_BUILD_HINT = ("Build it with `bash run_scripts/build_utonia_pointrope.sh` "
               "before using the CUDA implementation.")


def _load_extension():
    global _EXT
    global _EXT_LOAD_ERROR
    if _EXT is not None or _EXT_LOAD_ERROR is not None:
        return _EXT
    try:
        try:
            _EXT = importlib.import_module(f"{__name__}._C")
        except Exception:
            _EXT = importlib.import_module("point3d_rope._C")
    except Exception as exc:  # pragma: no cover - environment dependent
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


def utonia_pointrope_available() -> bool:
    return _load_extension() is not None


def utonia_pointrope_build_info() -> dict[str, object]:
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
        raise RuntimeError(
            f"Utonia PointRoPE CUDA extension is unavailable.{detail} {_BUILD_HINT}"
        )
    return ext


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
        raise TypeError("q and k must be torch.Tensor instances.")
    if not isinstance(positions, torch.Tensor):
        raise TypeError("positions must be a torch.Tensor.")
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(
            f"q and k must have shape (N,H,D), got {tuple(q.shape)} and {tuple(k.shape)}."
        )
    if tuple(q.shape) != tuple(k.shape):
        raise ValueError(
            f"q and k must have identical shapes, got {tuple(q.shape)} and {tuple(k.shape)}."
        )
    if positions.ndim != 2 or tuple(positions.shape) != (int(q.shape[0]), 3):
        raise ValueError(
            f"positions must have shape ({int(q.shape[0])},3), got {tuple(positions.shape)}."
        )
    if q.device.type != "cuda" or k.device.type != "cuda" or positions.device.type != "cuda":
        raise ValueError("q, k, and positions must be CUDA tensors.")
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError(
            f"q/k dtype must be float32, float16, or bfloat16, got {q.dtype}.")
    if k.dtype != q.dtype:
        raise TypeError(
            f"q and k must have identical dtypes, got {q.dtype} and {k.dtype}."
        )
    if positions.dtype != torch.float32:
        positions = positions.float()
    return q.contiguous(), k.contiguous(), positions.contiguous()


class _UtoniaPointRoPEFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        base: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, positions = _validate_inputs(q, k, positions)
        q_out, k_out = _require_extension().apply(q, k, positions, float(base),
                                                  1.0)
        ctx.save_for_backward(positions)
        ctx.base = float(base)
        return q_out, k_out

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        (positions, ) = ctx.saved_tensors
        if grad_q is None and grad_k is None:
            return None, None, None, None
        if grad_q is None:
            grad_q_input = torch.zeros_like(grad_k)
        else:
            grad_q_input = grad_q
        if grad_k is None:
            grad_k_input = torch.zeros_like(grad_q)
        else:
            grad_k_input = grad_k
        grad_q_rot, grad_k_rot = _require_extension().apply(
            grad_q_input.contiguous(),
            grad_k_input.contiguous(),
            positions,
            ctx.base,
            -1.0,
        )
        grad_q_out = grad_q_rot if grad_q is not None else None
        grad_k_out = grad_k_rot if grad_k is not None else None
        return grad_q_out, grad_k_out, None, None


def utonia_pointrope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    *,
    base: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Utonia's continuous-coordinate 3D RoPE to q and k.

    Inputs use the Utonia attention layout (N, heads, head_dim). The head
    dimension is split into three coordinate chunks, each with paired
    rotation halves.
    """
    if not isinstance(base, (float, int)) or float(base) <= 0:
        raise ValueError(f"base must be positive, got {base!r}.")
    return _UtoniaPointRoPEFunction.apply(q, k, positions, float(base))


_SUPPORTED_BACKENDS = {"torch", "cuda", "auto"}


def configured_utonia_pointrope_backend() -> str:
    backend = os.environ.get("UTONIA_POINTROPE_BACKEND",
                             "torch").strip().lower()
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            "UTONIA_POINTROPE_BACKEND must be one of {'torch', 'cuda', 'auto'}; "
            f"got {backend!r}")
    return backend


def resolved_utonia_pointrope_backend() -> str:
    backend = configured_utonia_pointrope_backend()
    if backend == "auto":
        return "cuda" if utonia_pointrope_available(
        ) and torch.cuda.is_available() else "torch"
    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "UTONIA_POINTROPE_BACKEND=cuda requested, but CUDA is unavailable."
            )
        if not utonia_pointrope_available():
            detail = (f" load_error={_EXT_LOAD_ERROR!r}"
                      if _EXT_LOAD_ERROR is not None else "")
            raise RuntimeError(
                "UTONIA_POINTROPE_BACKEND=cuda requested, but the CUDA extension "
                f"is unavailable.{detail} {_BUILD_HINT}")
    return backend


class UtoniaPointRoPEAdapter(nn.Module):
    """State-compatible adapter from Utonia's torch RoPE to the CUDA kernel."""

    def __init__(self, reference_rope: nn.Module):
        super().__init__()
        if not hasattr(reference_rope, "inv_freq"):
            raise ValueError(
                "Utonia Point3DRoPE must expose an inv_freq buffer.")
        self.head_dim = int(reference_rope.head_dim)
        self.chunk_dim = int(reference_rope.chunk_dim)
        self.base = float(reference_rope.base)
        # Keep the original state_dict key for strict resume loading. The CUDA
        # kernel computes the same frequencies internally and does not read it.
        self.register_buffer("inv_freq",
                             reference_rope.inv_freq.detach().clone())

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return utonia_pointrope(q, k, xyz, base=self.base)


def patch_utonia_pointrope_backend(model: nn.Module) -> tuple[str, int]:
    """Select Utonia's RoPE backend without changing model parameter names."""
    backend = resolved_utonia_pointrope_backend()
    if backend == "torch":
        return backend, 0

    replaced = 0
    for module in model.modules():
        if module.__class__.__name__ != "SerializedAttention":
            continue
        rope = getattr(module, "rope", None)
        if isinstance(rope, UtoniaPointRoPEAdapter):
            replaced += 1
            continue
        if rope is None:
            raise RuntimeError(
                "SerializedAttention module is missing its rope submodule.")
        module.rope = UtoniaPointRoPEAdapter(rope)
        replaced += 1

    if replaced == 0:
        raise RuntimeError(
            "No Utonia SerializedAttention.rope modules were found.")
    return backend, replaced


__all__ = [
    "utonia_pointrope",
    "utonia_pointrope_available",
    "utonia_pointrope_build_info",
    "configured_utonia_pointrope_backend",
    "resolved_utonia_pointrope_backend",
    "UtoniaPointRoPEAdapter",
    "patch_utonia_pointrope_backend",
]
