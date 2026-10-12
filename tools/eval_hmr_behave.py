#!/usr/bin/env python3
"""Evaluate a Pointcept human-pose checkpoint on the BEHAVE eval shard."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "exp/human_pose/hmr-utonia-v1m1-0-bedlam/config.py"
DEFAULT_WEIGHT = ROOT / "exp/human_pose/hmr-utonia-v1m1-0-bedlam/model/model_best.pth"
DEFAULT_BEHAVE_ROOT = Path(
    "/home/juyiang/data/PointTransformerV3_HumanPointCloud/data/behave/"
    "behave_evaluation_full_shard"
)
DEFAULT_SMPL_ROOT = Path("/home/juyiang/data/SMPL_models/models")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a Pointcept human-pose checkpoint on BEHAVE."
    )
    parser.add_argument(
        "--config-file",
        "--config",
        type=Path,
        default=Path(os.environ.get("EVAL_CONFIG", str(DEFAULT_CONFIG))),
        help="Pointcept config; defaults to the current human-pose experiment.",
    )
    parser.add_argument(
        "--weight",
        type=Path,
        default=Path(os.environ.get("EVAL_WEIGHT", str(DEFAULT_WEIGHT))),
        help="Checkpoint to evaluate.",
    )
    parser.add_argument(
        "--behave-root",
        type=Path,
        default=Path(os.environ.get("BEHAVE_EVAL_SEQ_ROOT", str(DEFAULT_BEHAVE_ROOT))),
        help="BEHAVE evaluation shard root.",
    )
    parser.add_argument(
        "--smpl-models-dir",
        type=Path,
        default=Path(os.environ.get("SMPL_MODELS_DIR", str(DEFAULT_SMPL_ROOT))),
        help="Directory containing the SMPL-X model files.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("EVAL_DEVICE", "cuda:0"),
        help="Torch device, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("EVAL_BATCH_SIZE", "64")),
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("EVAL_NUM_WORKERS", "4")),
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample limit for a quick smoke evaluation.",
    )
    return parser


def _load_checkpoint(model: torch.nn.Module, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint has no state dict: {path}")
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if not key.removeprefix("module.").startswith("_smpl_runner.")
    }
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint does not match the configured model: "
            f"missing={incompatible.missing_keys[:10]} "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )


def _move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _autocast_context(device: torch.device, enabled: bool, dtype: str):
    if not enabled or device.type not in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, enabled=False)
    amp_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(str(dtype).lower(), torch.bfloat16)
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _build_dataset(cfg, args: argparse.Namespace):
    from pointcept.datasets import build_dataset
    from pointcept.datasets.human_pose import human_pose_collate_fn

    dataset_cfg = copy.deepcopy(cfg.data.val)
    dataset_cfg.manifest = str(args.behave_root.expanduser().resolve())
    dataset_cfg.smpl_models_dir = str(args.smpl_models_dir.expanduser().resolve())
    dataset_cfg.training = False
    dataset_cfg.human_pose_collate = True
    if args.max_samples is not None:
        dataset_cfg.max_samples = int(args.max_samples)
    return build_dataset(dataset_cfg), human_pose_collate_fn


def evaluate(args: argparse.Namespace) -> dict[str, float | int]:
    from pointcept.utils.config import Config
    from pointcept.utils.human_pose_metrics import (
        compute_mpjpe_pa_mpjpe_mm,
        compute_pve_mm,
    )

    config_path = args.config_file.expanduser().resolve()
    weight_path = args.weight.expanduser().resolve()
    behave_root = args.behave_root.expanduser().resolve()
    smpl_models_dir = args.smpl_models_dir.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not behave_root.is_dir():
        raise FileNotFoundError(f"BEHAVE shard root not found: {behave_root}")
    if not smpl_models_dir.is_dir():
        raise FileNotFoundError(f"SMPL-X model directory not found: {smpl_models_dir}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    cfg = Config.fromfile(str(config_path))
    cfg.model.smpl.models_dir = str(smpl_models_dir)
    from pointcept.models import build_model

    model = build_model(cfg.model)
    _load_checkpoint(model, weight_path)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for the requested device.")
        torch.cuda.set_device(device)
    model.to(device).eval()

    dataset, collate_fn = _build_dataset(cfg, args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    amp_enabled = bool(getattr(cfg, "enable_amp", False)) and device.type == "cuda"
    amp_dtype = str(getattr(cfg, "amp_dtype", "bfloat16"))
    metric_sums = {"mpjpe": 0.0, "pa_mpjpe": 0.0, "pve": 0.0}
    sample_count = 0
    point_count = 0
    start_time = time.time()
    print(
        f"[eval] checkpoint={weight_path} samples={len(dataset)} "
        f"batch_size={args.batch_size} device={device}",
        flush=True,
    )
    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = _move_to_device(batch, device)
            target_joints = batch["gt_joints_cam"][:, :22]
            model_input = dict(batch)
            # Avoid computing training losses during standalone inference. The
            # model still creates pred_vertices/pred_joints_cam through SMPL-X.
            model_input.pop("gt_pose_body_aa", None)
            with _autocast_context(device, amp_enabled, amp_dtype):
                output = model(model_input)
            pred_joints = output.get("pred_joints_cam")
            pred_vertices = output.get("pred_vertices")
            if pred_joints is None or pred_vertices is None:
                raise RuntimeError(
                    "The model did not return SMPL-X predictions. Check the SMPL-X "
                    "assets and model configuration."
                )
            gt_geometry = model._smpl_gt_geometry(batch)
            if gt_geometry is None:
                raise RuntimeError("Unable to build SMPL-X ground-truth geometry.")
            gt_vertices = gt_geometry[0]
            mpjpe, pa_mpjpe = compute_mpjpe_pa_mpjpe_mm(
                pred_joints[:, :22], target_joints
            )
            pve = compute_pve_mm(pred_vertices, gt_vertices)
            metric_sums["mpjpe"] += float(mpjpe.sum())
            metric_sums["pa_mpjpe"] += float(pa_mpjpe.sum())
            metric_sums["pve"] += float(pve.sum())
            sample_count += int(pred_joints.shape[0])
            point_count += int(batch["coord"].shape[0])
            if batch_idx == 1 or batch_idx % 20 == 0 or batch_idx == len(loader):
                print(
                    f"[eval] batch={batch_idx}/{len(loader)} samples={sample_count} "
                    f"mpjpe={float(mpjpe.mean()):.3f} "
                    f"pa_mpjpe={float(pa_mpjpe.mean()):.3f} "
                    f"pve={float(pve.mean()):.3f}",
                    flush=True,
                )
    if sample_count == 0:
        raise RuntimeError("No BEHAVE samples were evaluated.")
    elapsed = time.time() - start_time
    result = {
        "mpjpe": metric_sums["mpjpe"] / sample_count,
        "pa_mpjpe": metric_sums["pa_mpjpe"] / sample_count,
        "pve": metric_sums["pve"] / sample_count,
        "samples": sample_count,
        "points": point_count,
        "seconds": elapsed,
    }
    print(
        f"BEHAVE: samples={sample_count} points={point_count} "
        f"MPJPE={result['mpjpe']:.3f} mm "
        f"PA-MPJPE={result['pa_mpjpe']:.3f} mm "
        f"PVE={result['pve']:.3f} mm "
        f"time={elapsed:.1f}s",
        flush=True,
    )
    return result


def main() -> int:
    evaluate(_build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
