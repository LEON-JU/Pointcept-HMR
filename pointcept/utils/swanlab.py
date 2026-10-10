"""Small SwanLab adapter used by Pointcept training and evaluation hooks."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any


_SWANLAB = None


def _plain_config(value):
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain_config(to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    return value


def _module():
    global _SWANLAB
    if _SWANLAB is None:
        try:
            _SWANLAB = importlib.import_module("swanlab")
        except ImportError as exc:
            raise RuntimeError(
                "SwanLab logging is enabled but the `swanlab` package is not "
                "installed in the active environment."
            ) from exc
    return _SWANLAB


def init(*, project, name, config, mode="online", log_dir=None):
    mode = {"cloud": "online"}.get(str(mode), str(mode))
    kwargs = {
        "project": str(project),
        "name": str(name),
        "config": _plain_config(config),
        "mode": mode,
    }
    if log_dir is not None:
        kwargs["log_dir"] = str(log_dir)
    return _module().init(**kwargs)


def log(data: dict[str, Any], *, step: int):
    _module().log(data, step=int(step))


def define_metric(*args, **kwargs):
    """SwanLab uses the explicit step passed to each log call."""
    del args, kwargs


def finish():
    finish_fn = getattr(_module(), "finish", None)
    if callable(finish_fn):
        finish_fn()
