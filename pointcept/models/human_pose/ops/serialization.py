from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import torch

_EXT = None
_EXT_LOAD_ERROR: Exception | None = None
_SOURCE_HASH: str | None = None

_PKG_DIR = Path(__file__).resolve().parent
_SOURCE_FILES = (
    _PKG_DIR / "bindings.cpp",
    _PKG_DIR / "hilbert_cuda.cu",
    _PKG_DIR / "hilbert_cuda_approx.cu",
    _PKG_DIR / "morton_cuda.cu",
)
_BUILD_HINT = "Build it once with `bash run_scripts/build_serialization_cuda.sh` before enabling the CUDA serialization backend."


def _load_extension():
    global _EXT
    global _EXT_LOAD_ERROR
    if _EXT is not None or _EXT_LOAD_ERROR is not None:
        return _EXT
    try:
        _EXT = importlib.import_module(f"{__name__}._C")
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


def serialization_cuda_available() -> bool:
    return _load_extension() is not None


def serialization_cuda_build_info() -> dict[str, object]:
    ext = _load_extension()
    info: dict[str, object] = {
        "available": ext is not None,
        "module_name": f"{__name__}._C",
        "module_path": str(Path(ext.__file__).resolve()) if ext is not None else None,
        "load_error": repr(_EXT_LOAD_ERROR) if _EXT_LOAD_ERROR is not None else None,
        "source_hash": _source_hash(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["current_device"] = int(torch.cuda.current_device())
        info["device_capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability())
    return info


def _require_extension():
    ext = _load_extension()
    if ext is None:
        detail = f" load_error={_EXT_LOAD_ERROR!r}" if _EXT_LOAD_ERROR is not None else ""
        raise RuntimeError(
            "Precompiled serialize_cuda is unavailable."
            f"{detail} {_BUILD_HINT}"
        )
    return ext


def _shape_str(tensor: torch.Tensor) -> str:
    return str(tuple(int(v) for v in tensor.shape))


def _require_cuda_coords(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)!r}.")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor, got device={tensor.device}.")
    if tensor.ndim != 2 or int(tensor.shape[1]) != 3:
        raise ValueError(f"{name} must have shape (N,3), got shape={_shape_str(tensor)}.")
    return tensor.contiguous()


def morton_encode(coords: torch.Tensor) -> torch.Tensor:
    ext = _require_extension()
    coords = _require_cuda_coords("coords", coords)
    return ext.morton_encode(coords).to(dtype=torch.long)


def hilbert_encode(coords: torch.Tensor, num_bits: int) -> torch.Tensor:
    ext = _require_extension()
    coords = _require_cuda_coords("coords", coords)
    return ext.hilbert_encode(coords, int(num_bits)).to(dtype=torch.long)


def hilbert_encode_approx(coords: torch.Tensor, num_bits: int) -> torch.Tensor:
    ext = _require_extension()
    coords = _require_cuda_coords("coords", coords)
    return ext.hilbert_encode_approx(coords, int(num_bits)).to(dtype=torch.long)


__all__ = [
    "hilbert_encode",
    "hilbert_encode_approx",
    "morton_encode",
    "serialization_cuda_available",
    "serialization_cuda_build_info",
]
