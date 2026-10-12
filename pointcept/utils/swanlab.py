"""Small SwanLab adapter used by Pointcept training and evaluation hooks."""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from pathlib import Path
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


def _resume_run_id(log_dir):
    if not log_dir:
        raise RuntimeError("SwanLab resume requires log_dir to locate the previous run ID.")

    run_id_file = Path(log_dir).parent / "swanlab_run_id"
    if run_id_file.is_file():
        run_id = run_id_file.read_text(encoding="utf-8").strip()
        if run_id:
            return run_id

    run_files = Path(log_dir).glob("run-*/run-*.swanlab")
    run_ids = {path.stem.removeprefix("run-") for path in run_files}
    if len(run_ids) == 1:
        return run_ids.pop()
    if not run_ids:
        raise RuntimeError(
            f"No previous SwanLab run ID found under {log_dir}; refusing to create a new run."
        )
    raise RuntimeError(
        f"Multiple SwanLab run IDs found under {log_dir}: {sorted(run_ids)}. "
        "Set SWANLAB_RUN_ID explicitly to select the run to resume."
    )


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

    resume_mode = os.environ.get("SWANLAB_RESUME", "").lower()
    if mode == "online" and resume_mode in {"true", "1"}:
        resume_mode = "allow"
    if mode == "online" and resume_mode in {"allow", "must"}:
        run_id = os.environ.get("SWANLAB_RUN_ID") or _resume_run_id(log_dir)
        kwargs.update(id=run_id, resume=resume_mode)

    run = _module().init(**kwargs)
    if mode == "online" and log_dir is not None:
        run_id = getattr(run, "id", None)
        if run_id:
            run_id_file = Path(log_dir).parent / "swanlab_run_id"
            # Keep the selected experiment stable if a launch accidentally omits resume.
            if resume_mode in {"allow", "must"} or not run_id_file.exists():
                temporary_file = run_id_file.with_name(run_id_file.name + ".tmp")
                temporary_file.write_text(str(run_id) + "\n", encoding="utf-8")
                temporary_file.replace(run_id_file)
    return run


def log(data: dict[str, Any], *, step: int):
    _module().log(data, step=int(step))


def define_metric(*args, **kwargs):
    """SwanLab uses the explicit step passed to each log call."""
    del args, kwargs


def finish():
    finish_fn = getattr(_module(), "finish", None)
    if callable(finish_fn):
        finish_fn()
