from __future__ import annotations

import json
import pickle
import os
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

from .builder import DATASETS
from pointcept.utils.human_pose_geometry import (
    axis_angle_to_matrix, matrix_to_axis_angle, smplx_rest_root,
    stabilize_axis_angle,
)


def _load_record(source):
    if isinstance(source, dict):
        return dict(source)
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        return dict(obj) if isinstance(obj, dict) else obj
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        return dict(obj)
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix == ".ply":
        lines = path.read_text().splitlines()
        end = lines.index("end_header")
        count = next(
            int(line.split()[-1]) for line in lines[:end]
            if line.startswith("element vertex"))
        xyz = np.asarray([[float(v) for v in line.split()[:3]]
                          for line in lines[end + 1:end + 1 + count]],
                         dtype=np.float32)
        return {"coord": xyz, "name": path.stem}
    raise ValueError(f"Unsupported human-pose sample format: {path}")


def _records_from_manifest(manifest):
    if manifest is None:
        return []
    path = Path(manifest)
    if path.is_dir():
        return sorted(str(p) for p in path.glob("*.npz")) + sorted(
            str(p) for p in path.glob("*.pt"))
    if path.suffix == ".jsonl":
        return [
            json.loads(line) for line in path.read_text().splitlines()
            if line.strip()
        ]
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, list) else payload.get(
            "samples", [])
    return [str(path)]


def _tensor(value, dtype=torch.float32):
    return torch.as_tensor(value, dtype=dtype)


def _filter_extreme_pointcloud_outliers(
        coord,
        color,
        normal,
        radius_quantile=0.95,
        radius_scale=1.0,
        min_radius_m=1.0,
        max_radius_m=1.0,
        min_keep_points=256):
    """Drop invalid/far points before centering and serialization."""
    coord = torch.as_tensor(coord, dtype=torch.float32)
    color = torch.as_tensor(color)
    normal = torch.as_tensor(normal, dtype=torch.float32)
    if coord.ndim != 2 or coord.shape[1] != 3:
        raise ValueError(f"coord must have shape (N, 3), got {tuple(coord.shape)}")
    n_points = int(coord.shape[0])
    if color.ndim != 2 or color.shape[0] != n_points:
        raise ValueError(
            f"color must have the same point count as coord, got {tuple(color.shape)}"
        )
    if normal.ndim != 2 or normal.shape != coord.shape:
        raise ValueError(
            f"normal must have shape {tuple(coord.shape)}, got {tuple(normal.shape)}"
        )

    keep = torch.isfinite(coord).all(dim=1)
    keep &= torch.isfinite(normal).all(dim=1)
    if color.is_floating_point():
        keep &= torch.isfinite(color).all(dim=1)
    valid = torch.nonzero(keep, as_tuple=False).flatten()
    if valid.numel() == 0:
        return coord[:0], color[:0], normal[:0]

    valid_coord = coord[valid]
    center = valid_coord.median(dim=0).values
    radius = torch.linalg.vector_norm(valid_coord - center, dim=1)
    quantile = torch.quantile(radius, float(radius_quantile))
    threshold = torch.clamp(
        quantile * float(radius_scale),
        min=float(min_radius_m),
        max=float(max_radius_m),
    )
    local_keep = radius <= threshold
    target = min(int(min_keep_points), int(valid.numel()))
    if int(local_keep.sum()) < target:
        nearest = torch.argsort(radius)[:target]
        local_keep = torch.zeros_like(local_keep, dtype=torch.bool)
        local_keep[nearest] = True
    selected = valid[local_keep]
    return coord[selected], color[selected], normal[selected]


class _HumanPoseDataset(Dataset):

    def __init__(self,
                 manifest=None,
                 min_points=32,
                 grid_size=0.01,
                 training=False,
                 max_samples=None,
                 loop=1,
                 **kwargs):
        self.augmentation = dict(kwargs.get("augmentation") or {})
        self._configure_outlier_filter(kwargs)
        self.records = _records_from_manifest(manifest)
        if max_samples is not None:
            self.records = self.records[:int(max_samples)]
        self.min_points, self.grid_size, self.training = int(
            min_points), float(grid_size), bool(training)
        self.loop = int(loop) if self.training else 1

    def _configure_outlier_filter(self, options):
        self.filter_extreme_outliers = bool(
            options.pop("filter_extreme_outliers", True))
        self.outlier_radius_quantile = float(
            options.pop("outlier_radius_quantile", 0.95))
        self.outlier_radius_scale = float(
            options.pop("outlier_radius_scale", 1.0))
        self.outlier_min_radius_m = float(
            options.pop("outlier_min_radius_m", 1.0))
        self.outlier_max_radius_m = float(
            options.pop("outlier_max_radius_m", 1.0))
        self.outlier_min_keep_points = int(
            options.pop("outlier_min_keep_points", 256))
        self.smpl_models_dir = str(options.pop(
            "smpl_models_dir",
            os.environ.get("SMPLX_MODELS_DIR", "/home/juyiang/data/SMPL_models/models")))
        del options

    def __len__(self):
        return len(self.records) * self.loop

    def _prepare(self, raw, index):
        raw = _load_record(raw)
        coord = _tensor(raw.get("coord", raw.get("points"))).clone()
        if coord.ndim != 2 or coord.shape[1] != 3:
            raise ValueError(
                f"coord must have shape (N,3), got {tuple(coord.shape)}")
        if coord.shape[0] < self.min_points:
            raise ValueError(
                f"sample {index} has only {coord.shape[0]} points")
        color = _tensor(raw.get("color", raw.get("rgb", np.zeros_like(coord))))
        normal = _tensor(raw.get("normal", np.zeros_like(coord))).clone()
        if self.filter_extreme_outliers:
            coord, color, normal = _filter_extreme_pointcloud_outliers(
                coord,
                color,
                normal,
                radius_quantile=self.outlier_radius_quantile,
                radius_scale=self.outlier_radius_scale,
                min_radius_m=self.outlier_min_radius_m,
                max_radius_m=self.outlier_max_radius_m,
                min_keep_points=self.outlier_min_keep_points,
            )
            if coord.shape[0] < self.min_points:
                raise ValueError(
                    f"sample {index} has only {coord.shape[0]} valid inlier points "
                    f"after outlier filtering (minimum {self.min_points})")
        if color.max() > 1.0: color = color / 255.0
        betas = (_tensor(raw["gt_betas"]).clone()
                 if "gt_betas" in raw else None)
        joints = (_tensor(raw["gt_joints_cam"]).clone()
                  if "gt_joints_cam" in raw else None)
        pose = (_tensor(raw["gt_pose_body_aa"]).clone()
                if "gt_pose_body_aa" in raw else None)
        trans = (_tensor(raw["gt_trans_cam"]).clone()
                 if "gt_trans_cam" in raw else None)
        prevoxelized = bool(raw.get("prevoxelized", False)) and not self.training
        rest_root = None
        if betas is not None and Path(self.smpl_models_dir, "smplx",
                                     "SMPLX_NEUTRAL.npz").is_file():
            rest_root = smplx_rest_root(betas.unsqueeze(0),
                                         self.smpl_models_dir)[0]
        if self.training:
            permutation = torch.tensor([
                0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17,
                16, 19, 18, 21, 20
            ])
            if np.random.random() < float(
                    getattr(self, "augmentation",
                            {}).get("flip_probability", 0.5)):
                coord[:, 0].neg_()
                normal[:, 0].neg_()
                if joints is not None:
                    joints[:, 0].neg_()
                    joints = joints[permutation]
                if trans is not None:
                    if rest_root is None:
                        trans[0].neg_()
                    else:
                        trans = trans.new_tensor([-1, 1, 1]) * (trans + rest_root) - rest_root
                if pose is not None:
                    pose = pose[permutation]
                    pose *= pose.new_tensor([1.0, -1.0, -1.0])
            rotation_center_mode = getattr(self, "augmentation", {}).get(
                "rotation_center", "root")
            if rotation_center_mode not in {"root", "point_cloud"}:
                raise ValueError(
                    "augmentation.rotation_center must be 'root' or 'point_cloud', "
                    f"got {rotation_center_mode!r}")
            if rotation_center_mode == "root" and trans is not None and rest_root is not None:
                rotation_center = trans + rest_root
            elif rotation_center_mode == "root" and joints is not None:
                rotation_center = joints[0].clone()
            else:
                rotation_center = coord.mean(0)
            rotation = torch.eye(3)
            for axis in range(3):
                if np.random.random() < float(
                        getattr(self, "augmentation",
                                {}).get("rotation_axis_probability", 0.5)):
                    angle = float(np.random.uniform(-np.pi, np.pi))
                    c, s = np.cos(angle), np.sin(angle)
                    one = torch.eye(3)
                    if axis == 0: one[1:, 1:] = torch.tensor([[c, -s], [s, c]])
                    elif axis == 1:
                        one[0, 0] = c
                        one[0, 2] = s
                        one[2, 0] = -s
                        one[2, 2] = c
                    else:
                        one[:2, :2] = torch.tensor([[c, -s], [s, c]])
                    rotation = one @ rotation
            coord = (coord - rotation_center) @ rotation.T + rotation_center
            normal = normal @ rotation.T
            if joints is not None:
                joints = (joints - rotation_center) @ rotation.T + rotation_center
            if trans is not None:
                if rest_root is None:
                    trans = (trans - rotation_center) @ rotation.T + rotation_center
                else:
                    trans = (trans + rest_root - rotation_center) @ rotation.T + rotation_center - rest_root
            if pose is not None:
                root_rot = rotation @ axis_angle_to_matrix(pose[0])
                pose[0] = stabilize_axis_angle(matrix_to_axis_angle(root_rot))
            augmentation = getattr(self, "augmentation", {})
            sigma = float(augmentation.get("jitter_sigma", 0.01))
            clip = float(augmentation.get("jitter_clip", 0.05))
            jitter_probability = float(
                augmentation.get("jitter_probability", 1.0))
            if np.random.random() < jitter_probability:
                coord += torch.randn_like(coord).mul_(sigma).clamp_(-clip, clip)
        centroid = (_tensor(raw["point_centroid"]).clone()
                    if prevoxelized and "point_centroid" in raw else coord.mean(0))
        # Shards contain metric camera coordinates; the model consumes the
        # same coordinates centered by the saved legacy centroid.
        coord = coord - centroid
        grid_coord = torch.floor(coord / self.grid_size).to(torch.int32)
        grid_coord -= grid_coord.min(0).values
        _, inverse = torch.unique(grid_coord, dim=0, sorted=True,
                                  return_inverse=True)
        groups = int(inverse.max().item()) + 1
        if prevoxelized:
            keep = torch.arange(coord.shape[0])
        elif self.training:
            order = torch.randperm(coord.shape[0])
            keep = torch.full((groups, ), -1, dtype=torch.long)
            for point_id in order.tolist():
                keep[inverse[point_id]] = point_id
            keep = keep[keep >= 0].sort().values
        else:
            keep = torch.full((groups, ), coord.shape[0], dtype=torch.long)
            ids = torch.arange(coord.shape[0])
            keep.scatter_reduce_(0,
                                 inverse,
                                 ids,
                                 reduce="amin",
                                 include_self=True)
            keep = keep.sort().values
        coord, color, normal, grid_coord = coord[keep], color[keep], normal[
            keep], grid_coord[keep]
        out = {
            "coord": coord.contiguous(),
            "color": color.contiguous(),
            "normal": normal.contiguous(),
            "feat": torch.cat([coord, color, normal], dim=1).contiguous(),
            "grid_coord": grid_coord,
            "point_centroid": centroid,
            "name": raw.get("name", str(index)),
        }
        defaults = {
            "gt_pose_body_aa": (22, 3),
            "gt_betas": (10, ),
            "gt_trans_cam": (3, ),
            "gt_trans_delta": (3, ),
            "gt_joints_cam": (22, 3)
        }
        for key, shape in defaults.items():
            if key in raw: out[key] = _tensor(raw[key])
            elif key in {
                    "gt_pose_body_aa", "gt_betas", "gt_trans_cam",
                    "gt_trans_delta", "gt_joints_cam"
            }:
                out[key] = torch.zeros(shape, dtype=torch.float32)
        if joints is not None: out["gt_joints_cam"] = joints
        if pose is not None: out["gt_pose_body_aa"] = pose
        if trans is not None: out["gt_trans_cam"] = trans
        if trans is not None: out["gt_trans_delta"] = trans - centroid
        for key in ("cam_intrinsic_matrix", "image_width", "image_height"):
            if key in raw: out[key] = _tensor(raw[key])
        return out

    def __getitem__(self, index):
        return self._prepare(self.records[index % len(self.records)], index)


@DATASETS.register_module("BedlamHumanPoseDataset")
class BedlamHumanPoseDataset(_HumanPoseDataset):

    def __init__(self,
                 pc_root="/home/juyiang/data/bedlam/pc_row_shard",
                 gt_root="/home/juyiang/data/bedlam/gt_shard",
                 manifest=None,
                 min_points=128,
                 grid_size=0.01,
                 training=False,
                 max_samples=None,
                 augmentation=None,
                 loop=1,
                 **kwargs):
        self._configure_outlier_filter(kwargs)
        self.pc_root, self.gt_root = Path(pc_root), Path(gt_root)
        if not (self.pc_root / "manifest.json").is_file() or not (
                self.gt_root / "manifest.json").is_file():
            raise FileNotFoundError(
                f"BEDLAM shard roots require manifest.json: pc={self.pc_root} gt={self.gt_root}"
            )
        self.pc_manifest = json.loads(
            (self.pc_root / "manifest.json").read_text())
        self.gt_manifest = json.loads(
            (self.gt_root / "manifest.json").read_text())
        with np.load(self.pc_root /
                     self.pc_manifest["sample_index_path"]) as index:
            self.pc_index = {key: index[key] for key in index.files}
        with np.load(self.gt_root /
                     self.gt_manifest["sample_index_path"]) as index:
            self.gt_index = {key: index[key] for key in index.files}
        if len(self.pc_index["scene_id"]) != len(self.gt_index["scene_id"]):
            raise ValueError("BEDLAM point and GT sample_index sizes differ")
        sample_count = len(self.pc_index["scene_id"])
        point_lengths = np.empty(sample_count, dtype=np.int32)
        shard_ids = self.pc_index["shard_id"]
        row_ids = self.pc_index["row_in_shard"]
        for shard in self.pc_manifest["shards"]:
            shard_id = int(shard["shard_id"])
            start = int(np.searchsorted(shard_ids, shard_id, side="left"))
            end = int(np.searchsorted(shard_ids, shard_id, side="right"))
            if start == end:
                continue
            sidecar_path = self.pc_root / shard["length_rel_path"]
            if not sidecar_path.is_file():
                raise FileNotFoundError(
                    f"BEDLAM point-length sidecar is missing: {sidecar_path}")
            lengths = np.load(sidecar_path,
                              mmap_mode="r",
                              allow_pickle=False)
            point_lengths[start:end] = lengths[row_ids[start:end]]
        self.records = np.flatnonzero(point_lengths >= int(min_points))
        if max_samples is not None:
            self.records = self.records[:int(max_samples)]
        self.min_points, self.grid_size, self.training = int(
            min_points), float(grid_size), bool(training)
        self.loop = int(loop) if self.training else 1
        self.augmentation = dict(augmentation or {})
        self._pc_maps, self._gt_maps = OrderedDict(), OrderedDict()

    def __len__(self):
        return len(self.records) * self.loop

    def _row(self, root, manifest, index, cache, sample_i):
        shard_id, row_id = int(index["shard_id"][sample_i]), int(
            index["row_in_shard"][sample_i])
        if shard_id not in cache:
            record = next(item for item in manifest["shards"]
                          if int(item["shard_id"]) == shard_id)
            cache[shard_id] = np.load(root / record["rel_path"],
                                      mmap_mode="r",
                                      allow_pickle=False)
            while len(cache) > 2:
                cache.popitem(last=False)
        cache.move_to_end(shard_id)
        return cache[shard_id][row_id]

    def __getitem__(self, item):
        i = int(self.records[item % len(self.records)])
        pc = self._row(self.pc_root, self.pc_manifest, self.pc_index,
                       self._pc_maps, i)
        gt = self._row(self.gt_root, self.gt_manifest, self.gt_index,
                       self._gt_maps, i)
        n = int(pc["length"])
        if n < self.min_points:
            raise ValueError(f"BEDLAM sample {i} has only {n} points")
        raw = {
            "coord":
            np.asarray(pc["coord"][:n]).copy(),
            "color":
            np.asarray(pc["rgb"][:n]).copy(),
            "normal":
            np.asarray(pc["normal"][:n]).copy(),
            "gt_pose_body_aa":
            np.asarray(gt["pose_cam"]).reshape(55, 3)[:22].copy(),
            "gt_betas":
            np.asarray(gt["betas"][:10]).copy(),
            "gt_trans_cam":
            np.asarray(gt["trans_cam"]).copy(),
            "gt_joints_cam":
            np.asarray(gt["joints_cam"]).copy(),
            "cam_intrinsic_matrix":
            np.asarray(gt["cam_int"]).copy(),
            "image_width":
            int(gt["image_width"]),
            "image_height":
            int(gt["image_height"]),
            "name":
            f"bedlam-{int(self.pc_index['scene_id'][i])}-{int(self.pc_index['seq_id'][i])}-{int(self.pc_index['frame_id'][i])}-{int(self.pc_index['person_id'][i])}",
        }
        return self._prepare(raw, i)


@DATASETS.register_module("BehaveHumanPoseDataset")
class BehaveHumanPoseDataset(_HumanPoseDataset):

    def __init__(
            self,
            manifest="/home/juyiang/data/PointTransformerV3_HumanPointCloud/data/behave/behave_evaluation_full_shard",
            min_points=32,
            grid_size=0.01,
            training=False,
            max_samples=None,
            loop=1,
            **kwargs):
        self._configure_outlier_filter(kwargs)
        self.root = Path(manifest)
        meta = json.loads((self.root / "manifest.json").read_text())
        with np.load(self.root / meta["sample_index_path"]) as index:
            self.index = {key: index[key] for key in index.files}
        self.scene_names = json.loads(
            (self.root / meta["scene_names_path"]).read_text())
        self.manifest = meta
        self.min_points, self.grid_size, self.training = int(
            min_points), float(grid_size), bool(training)
        self.loop = int(loop) if self.training else 1
        count = len(self.index["scene_id"])
        self.records = np.arange(
            count if max_samples is None else min(count, int(max_samples)),
            dtype=np.int64)
        self._maps = OrderedDict()

    def __getitem__(self, item):
        i = int(self.records[item])
        shard_id = int(self.index["shard_id"][i])
        row_id = int(self.index["row_in_shard"][i])
        if shard_id not in self._maps:
            rec = next(r for r in self.manifest["shards"]
                       if int(r["shard_id"]) == shard_id)
            self._maps[shard_id] = np.load(self.root / rec["rel_path"],
                                           mmap_mode="r",
                                           allow_pickle=False)
            while len(self._maps) > 2:
                self._maps.popitem(last=False)
        self._maps.move_to_end(shard_id)
        row = self._maps[shard_id][row_id]
        n = int(row["length"])
        if n < self.min_points:
            raise ValueError(f"BEHAVE sample {i} has only {n} points")
        raw = {
            "coord":
            np.asarray(row["coord"][:n]).copy(),
            "color":
            np.asarray(row["rgb"][:n]).copy(),
            "normal":
            np.asarray(row["normal"][:n]).copy(),
            "point_centroid":
            np.asarray(row["point_centroid"]).copy(),
            "prevoxelized": True,
            "gt_pose_body_aa":
            np.asarray(row["gt_pose_body_aa"]).copy(),
            "gt_betas":
            np.asarray(row["gt_betas"]).copy(),
            "gt_trans_cam":
            np.asarray(row["gt_trans_cam"]).copy(),
            "gt_joints_cam":
            np.asarray(row["gt_joints_cam"]).copy(),
            "name":
            f"{self.scene_names[int(row['scene_id'])]}-k{int(row['kid_id'])}-{int(row['frame_id'])}"
        }
        # BEHAVE shards already store grid-sampled metric points and their original centroid.
        out = self._prepare(raw, i)
        out["point_centroid"] = _tensor(raw["point_centroid"])
        out["gt_trans_delta"] = out["gt_trans_cam"] - out["point_centroid"]
        return out


@DATASETS.register_module("SyntheticHumanPoseDataset")
class SyntheticHumanPoseDataset(_HumanPoseDataset):

    def __init__(self, length=8, points=128, seed=0, **kwargs):
        super().__init__(manifest=None, min_points=1, **kwargs)
        gen = torch.Generator().manual_seed(int(seed))
        self.samples = []
        for index in range(int(length)):
            coord = torch.randn((int(points), 3), generator=gen) * 0.5
            self.samples.append({
                "coord":
                coord,
                "color":
                torch.rand((int(points), 3), generator=gen),
                "normal":
                torch.randn((int(points), 3), generator=gen),
                "name":
                f"synthetic-{index}"
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self._prepare(self.samples[index], index)


class HumanPoseDistributedEvalSampler(Sampler):
    """Partition evaluation rows without padding or duplicating samples."""

    def __init__(self, dataset, rank=None, world_size=None):
        self.dataset = dataset
        self.rank = dist.get_rank(
        ) if rank is None and dist.is_initialized() else int(rank or 0)
        self.world_size = dist.get_world_size(
        ) if world_size is None and dist.is_initialized() else int(world_size
                                                                   or 1)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return max(0, (len(self.dataset) - self.rank + self.world_size - 1) //
                   self.world_size)


def human_pose_collate_fn(batch):
    if not batch: raise ValueError("cannot collate an empty batch")
    out = {}
    offsets, total = [], 0
    for item in batch:
        total += int(item["coord"].shape[0])
        offsets.append(total)
    for key in ("coord", "color", "normal", "feat", "grid_coord"):
        out[key] = torch.cat([item[key] for item in batch], 0)
    out["offset"] = torch.tensor(offsets, dtype=torch.long)
    for key in ("point_centroid", "gt_pose_body_aa", "gt_betas",
                "gt_trans_cam", "gt_trans_delta", "gt_joints_cam",
                "cam_intrinsic_matrix", "image_width", "image_height"):
        if key in batch[0]:
            out[key] = torch.stack([item[key] for item in batch])
    out["name"] = [item.get("name", str(i)) for i, item in enumerate(batch)]
    return out


__all__ = [
    "BedlamHumanPoseDataset", "BehaveHumanPoseDataset",
    "SyntheticHumanPoseDataset", "HumanPoseDistributedEvalSampler",
    "human_pose_collate_fn"
]
