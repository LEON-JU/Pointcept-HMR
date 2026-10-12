from __future__ import annotations

from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcept.models.builder import MODELS, build_model
from pointcept.models.utils.structure import Point
from .ops.point3d_rope import patch_utonia_pointrope_backend
from .ops.pointnet2 import furthest_point_sample_packed, gather_points, pointnet2_ext_available
from .ops.serialization import serialization_cuda_available
from .pointnet2_fusion import PointNetPPGlobalFusion
from pointcept.utils.human_pose_geometry import (
    axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_rot6d,
    rot6d_to_matrix, stabilize_axis_angle,
)


def _counts(offset):
    return torch.diff(torch.cat([offset.new_zeros(1), offset]))


_axis_angle_to_matrix = axis_angle_to_matrix
_matrix_to_rot6d = matrix_to_rot6d


class _QueryDecoderLayer(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim,
                                               heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim,
                                                heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(
            dim), nn.LayerNorm(dim)

    def forward(self, query, memory, mask):
        x, _ = self.self_attn(query, query, query, need_weights=False)
        query = self.norm1(query + x)
        x, weights = self.cross_attn(query,
                                     memory,
                                     memory,
                                     key_padding_mask=mask,
                                     need_weights=True)
        query = self.norm2(query + x)
        return self.norm3(query + self.ffn(query)), weights


@MODELS.register_module("UtoniaHumanPose")
class UtoniaHumanPose(nn.Module):
    """Pointcept Utonia encoder followed by the RGBD-HMR body regression head."""

    def __init__(
        self,
        backbone,
        backbone_out_channels=576,
        ops=None,
        decoder=None,
        smpl=None,
        loss=None,
        max_point_tokens=256,
        hidden_dim=256,
        num_heads=8,
        dropout=0.1,
        pretrained_path=None,
        require_smpl=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        if pretrained_path:
            checkpoint = torch.load(str(pretrained_path),
                                    map_location="cpu",
                                    weights_only=False)
            state = checkpoint.get("state_dict",
                                   checkpoint.get("model", checkpoint))
            state = {
                key.removeprefix("module."): value
                for key, value in state.items()
            }
            state.pop("embedding.mask_token", None)
            incompatible = self.backbone.load_state_dict(state, strict=False)
            missing = [
                key for key in incompatible.missing_keys
                if not key.endswith("mask_token")
            ]
            unexpected = list(incompatible.unexpected_keys)
            print(
                f"[UtoniaHumanPose] pretrained backbone={pretrained_path} missing={len(missing)} unexpected={len(unexpected)}"
            )
            if missing or unexpected:
                raise RuntimeError(
                    f"Utonia pretrained checkpoint mismatch: missing={missing[:20]} unexpected={unexpected[:20]}"
                )
        self.use_lora = bool(use_lora)
        if self.use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise RuntimeError(
                    "Utonia LoRA fine-tuning requires a compatible peft/"
                    f"transformers installation: {exc}"
                ) from exc

            lora_config = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                target_modules=["qkv"],
                lora_dropout=float(lora_dropout),
                bias="none",
            )
            # The raw Utonia checkpoint must be loaded before PEFT changes
            # the encoder parameter names.
            self.backbone.enc = get_peft_model(self.backbone.enc,
                                               lora_config)
            # Keep the whole original encoder fixed. The regression head and
            # the newly initialized LoRA adapters remain trainable.
            for parameter in self.backbone.embedding.parameters():
                parameter.requires_grad = False
            for name, parameter in self.backbone.enc.named_parameters():
                parameter.requires_grad = "lora_" in name
            self.backbone.enc.print_trainable_parameters()
        self.backbone_out_channels = int(backbone_out_channels)
        self.max_point_tokens = int(max_point_tokens)
        self.hidden_dim = int(hidden_dim)
        self.loss_weights = {
            "joint_2d": 10.0,
            "trans": 5.0,
            "pose": 0.5,
            "beta": 0.001,
            "cr_vert": 1.0,
            "smpl_joint": 1.0,
            **(loss or {}),
        }
        ops = dict(ops or {})
        self.ops = ops
        self._configure_ops()
        self.global_fusion = PointNetPPGlobalFusion(
            self.backbone_out_channels,
            backend=self.ops.get("pointnet2_backend", "torch"),
            max_centers=int((decoder or {}).get("global_centers", 64)),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.backbone_out_channels),
            nn.Linear(self.backbone_out_channels, hidden_dim - 3), nn.GELU())
        self.joint_seed = nn.Sequential(nn.Linear(1024, 22 * 64), nn.ReLU())
        self.joint_up_proj = nn.Linear(64, hidden_dim)
        # The decoder contains one query for each of the 22 body joints.
        self.decoder = nn.ModuleList([
            _QueryDecoderLayer(hidden_dim, num_heads, dropout)
            for _ in range(2)
        ])
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 6),
        )
        self.joint_token_reduce = nn.Sequential(nn.Linear(hidden_dim, 64),
                                                nn.ReLU(), nn.Dropout(dropout),
                                                nn.Linear(64, 32))
        self.trans_head = nn.Sequential(nn.Linear(22 * 32, 512), nn.ReLU(),
                                        nn.Linear(512, 256), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(256, 3))
        self.beta_pool_mlp = nn.Sequential(nn.Linear(hidden_dim, 256),
                                           nn.ReLU(), nn.Linear(256, 128),
                                           nn.Dropout(dropout), nn.ReLU())
        self.beta_head = nn.Linear(128, 10)
        self.smpl_cfg = dict(smpl or {})
        self._smpl_runner = None
        verts437_path = self.smpl_cfg.get("verts437_path")
        if verts437_path and Path(str(verts437_path)).is_file():
            ids = torch.load(str(verts437_path),
                             map_location="cpu",
                             weights_only=True).long().reshape(-1)
        else:
            ids = torch.empty(0, dtype=torch.long)
        self.register_buffer("verts437_ids", ids, persistent=False)
        if require_smpl:
            models_dir = Path(str(self.smpl_cfg.get("models_dir", "")))
            if not models_dir.is_dir():
                raise FileNotFoundError(
                    f"SMPL-X models_dir does not exist: {models_dir}")
            if self.verts437_ids.numel() == 0:
                raise FileNotFoundError(
                    f"SMPL-X verts437 asset is missing: {verts437_path}")
            try:
                import smplx  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "SMPL-X training requires the smplx package") from exc

    def _configure_ops(self):
        rope_backend = self.ops.get("point3d_rope_backend", "torch")
        serialization_backend = self.ops.get("serialization_backend", "torch")
        pointnet_backend = self.ops.get("pointnet2_backend", "torch")
        if rope_backend == "cuda":
            import os
            old = os.environ.get("UTONIA_POINTROPE_BACKEND")
            os.environ["UTONIA_POINTROPE_BACKEND"] = "cuda"
            try:
                backend, count = patch_utonia_pointrope_backend(self.backbone)
            finally:
                if old is None:
                    os.environ.pop("UTONIA_POINTROPE_BACKEND", None)
                else:
                    os.environ["UTONIA_POINTROPE_BACKEND"] = old
            if backend != "cuda" or count <= 0:
                raise RuntimeError(
                    "point3d_rope_backend='cuda' did not replace any Utonia RoPE modules"
                )
        if serialization_backend == "cuda" and not serialization_cuda_available(
        ):
            raise RuntimeError(
                "serialization_backend='cuda' requested but Pointcept serialization extension is unavailable"
            )
        if serialization_backend == "cuda":
            import os
            os.environ["POINTCEPT_SERIALIZATION_BACKEND"] = "cuda"
        if pointnet_backend == "cuda" and not pointnet2_ext_available():
            raise RuntimeError(
                "pointnet2_backend='cuda' requested but PointNet2 extension is unavailable"
            )

    def _deep_point(self, data):
        point = self.backbone(data)
        while isinstance(point, Point) and "pooling_parent" in point:
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent
        if not hasattr(point, "coord") or not hasattr(point, "feat"):
            raise RuntimeError(
                "Utonia backbone must return a Point with coord and feat")
        return point

    @staticmethod
    def _fps_reference(coord, offset, npoint):
        indices = []
        start = 0
        for end in offset.tolist():
            end = int(end)
            xyz = coord[start:end]
            if xyz.shape[0] == 0:
                raise ValueError(
                    "human pose batches cannot contain empty point segments")
            chosen = torch.zeros(npoint, dtype=torch.long, device=coord.device)
            distances = torch.full((xyz.shape[0], ),
                                   float("inf"),
                                   device=coord.device)
            last = 0
            for i in range(npoint):
                chosen[i] = last
                distances = torch.minimum(distances,
                                          ((xyz - xyz[last])**2).sum(-1))
                last = int(distances.argmax())
            indices.append(chosen + start)
            start = end
        return torch.cat(indices)

    def _memory(self, point):
        coord, feat, offset = point.coord.float(
        ), point.feat, point.offset.long()
        n = self.max_point_tokens
        if self.ops.get("pointnet2_backend", "torch") == "cuda":
            idx, _ = furthest_point_sample_packed(coord.contiguous(),
                                                  offset.contiguous(), n)
            sampled_coord, sampled_feat = gather_points(coord,
                                                        idx), gather_points(
                                                            feat.float(), idx)
        else:
            idx = self._fps_reference(coord, offset, n)
            sampled_coord, sampled_feat = coord[idx], feat.float()[idx]
        B = int(offset.numel())
        counts = _counts(offset).clamp(max=n)
        mask = torch.arange(
            n, device=coord.device).unsqueeze(0) >= counts.unsqueeze(1)
        memory = torch.cat(
            [sampled_coord, self.token_proj(sampled_feat)],
            dim=-1).view(B, n, self.hidden_dim)
        return memory, mask

    def _predict(self, data):
        point = self._deep_point(data)
        offset = point.offset.long()
        global_feat = self.global_fusion(point.coord, point.feat, offset)
        memory, mask = self._memory(point)
        query = self.joint_seed(global_feat).view(-1, 22, 64)
        query = self.joint_up_proj(query)
        weights = None
        for layer in self.decoder:
            query, weights = layer(query, memory, mask)
        pose6d = self.pose_head(query[:, :22])
        reduced = self.joint_token_reduce(query[:, :22])
        trans = self.trans_head(reduced.reshape(query.shape[0], -1))
        betas = self.beta_head(self.beta_pool_mlp(query[:, :22].amax(1)))
        output = {
            "pred_pose6d": pose6d,
            "pred_trans_delta": trans,
            "pred_betas": betas
        }
        smpl_output = self._smpl_forward(output, data)
        if smpl_output is not None:
            output.update(smpl_output)
        return output

    def _smpl_forward(self, pred, data):
        models_dir = self.smpl_cfg.get("models_dir")
        if not models_dir or str(models_dir).startswith("${") or not Path(
                str(models_dir)).is_dir():
            return None
        try:
            import smplx
        except ImportError as exc:
            raise RuntimeError(
                "SMPL-X models_dir is configured but the smplx package is unavailable"
            ) from exc
        if self._smpl_runner is None:
            self._smpl_runner = smplx.create(
                model_path=str(models_dir),
                model_type="smplx",
                gender=self.smpl_cfg.get("gender", "neutral"),
                ext="npz",
                num_betas=10,
                use_pca=False,
                flat_hand_mean=True,
            ).to(pred["pred_betas"].device).eval()
            self._smpl_runner.requires_grad_(False)
        pose_mat = pred["pred_pose6d"].reshape(-1, 22, 6).float()
        pose_aa = stabilize_axis_angle(
            matrix_to_axis_angle(rot6d_to_matrix(pose_mat))).reshape(-1, 22, 3)
        B = pose_aa.shape[0]
        zeros = pose_aa.new_zeros((B, 3))
        full_hand = pose_aa.new_zeros((B, 30, 3))
        centroid = data.get("point_centroid", pose_aa.new_zeros(
            (B, 3))).float()
        transl = pred["pred_trans_delta"].float() + centroid
        with torch.amp.autocast(device_type=pose_aa.device.type,
                                enabled=False) if pose_aa.device.type in {
                                    "cuda", "cpu"
                                } else nullcontext():
            result = self._smpl_runner(
                betas=pred["pred_betas"].float(),
                expression=pose_aa.new_zeros((B, 10)),
                global_orient=pose_aa[:, 0],
                body_pose=pose_aa[:, 1:22].reshape(B, 63),
                jaw_pose=zeros,
                leye_pose=zeros,
                reye_pose=zeros,
                left_hand_pose=full_hand[:, :15].reshape(B, 45),
                right_hand_pose=full_hand[:, 15:].reshape(B, 45),
                transl=transl,
                return_verts=True,
            )
        return {
            "pred_vertices": result.vertices,
            "pred_joints_cam": result.joints[:, :22]
        }

    def _smpl_gt_geometry(self, data):
        if self._smpl_runner is None or "gt_pose_body_aa" not in data or "gt_betas" not in data:
            return None
        pose = stabilize_axis_angle(data["gt_pose_body_aa"].float())
        B = pose.shape[0]
        hands = pose.new_zeros((B, 30, 3))
        zeros = pose.new_zeros((B, 3))
        transl = data.get("gt_trans_cam", zeros).float()
        with torch.no_grad(), torch.amp.autocast(device_type=pose.device.type,
                                                 enabled=False):
            result = self._smpl_runner(
                betas=data["gt_betas"].float(),
                expression=pose.new_zeros((B, 10)),
                global_orient=pose[:, 0],
                body_pose=pose[:, 1:22].reshape(B, 63),
                jaw_pose=zeros,
                leye_pose=zeros,
                reye_pose=zeros,
                left_hand_pose=hands[:, :15].reshape(B, 45),
                right_hand_pose=hands[:, 15:].reshape(B, 45),
                transl=transl,
                return_verts=True,
            )
        return result.vertices, result.joints[:, :22]

    def _loss(self, pred, data):
        device = pred["pred_trans_delta"].device
        zero = pred["pred_trans_delta"].new_zeros(())
        terms = {}
        gt_geometry = self._smpl_gt_geometry(data)
        gt_joints = (gt_geometry[1] if gt_geometry is not None else
                     data.get("gt_joints_cam"))
        if "gt_pose_body_aa" in data:
            gt = data["gt_pose_body_aa"].float()
            terms["pose"] = (
                pred["pred_pose6d"] -
                _matrix_to_rot6d(_axis_angle_to_matrix(gt))).square().mean()
        else:
            terms["pose"] = zero
        gt_trans = data.get("gt_trans_delta")
        terms["trans"] = (
            pred["pred_trans_delta"] -
            gt_trans.float()).square().mean() if gt_trans is not None else zero
        gt_beta = data.get("gt_betas")
        terms["beta"] = (
            pred["pred_betas"] -
            gt_beta.float()).square().mean() if gt_beta is not None else zero
        terms["joint_2d"] = zero
        if "pred_joints_cam" in pred and gt_joints is not None:
            pred_joints = pred["pred_joints_cam"]
            gt_joints = gt_joints.float()
            pred_root = pred_joints[:, :1]
            gt_root = gt_joints[:, :1]
            terms["smpl_joint"] = (
                pred_joints[:, :22] - pred_root -
                (gt_joints[:, :22] - gt_root)).square().mean()
            out_mpjpe = torch.linalg.norm(pred_joints[:, :22] -
                                          gt_joints[:, :22],
                                          dim=-1).mean()
        else:
            terms["smpl_joint"] = zero
            out_mpjpe = zero
        if "cam_intrinsic_matrix" in data and "pred_joints_cam" in pred and gt_joints is not None:
            K = data["cam_intrinsic_matrix"].float()
            gt_joints = gt_joints.float()[:, :22]
            pred_joints = pred["pred_joints_cam"][:, :22].float()
            valid = (pred_joints[..., 2] > 0) & (gt_joints[..., 2] > 0)

            def project(x):
                z = x[..., 2].clamp_min(1e-4)
                u = K[:, None, 0, 0] * x[..., 0] / z + K[:, None, 0, 2]
                v = K[:, None, 1, 1] * x[..., 1] / z + K[:, None, 1, 2]
                width = data["image_width"].float().reshape(-1, 1).clamp_min(1)
                height = data["image_height"].float().reshape(-1,
                                                              1).clamp_min(1)
                return torch.stack([u / width, v / height], -1)

            projected_pred = project(pred_joints)
            projected_gt = project(gt_joints)
            projected_pred = torch.where(valid[..., None], projected_pred,
                                         torch.zeros_like(projected_pred))
            projected_gt = torch.where(valid[..., None], projected_gt,
                                       torch.zeros_like(projected_gt))
            error = (projected_pred - projected_gt).square().sum(-1)
            count = valid.sum(1)
            per_sample = (error * valid).sum(1) / (2 * count.clamp_min(1))
            terms["joint_2d"] = torch.where(
                count > 0, per_sample, torch.zeros_like(per_sample)).mean()
        terms["cr_vert"] = zero
        if gt_geometry is not None and self.verts437_ids.numel(
        ) and "pred_vertices" in pred:
            pred_vertices = pred["pred_vertices"].index_select(
                1, self.verts437_ids)
            gt_vertices = gt_geometry[0].index_select(1, self.verts437_ids)
            pred_root = pred["pred_joints_cam"][:, :1]
            gt_root = gt_geometry[1][:, :1]
            terms["cr_vert"] = (pred_vertices - pred_root -
                                (gt_vertices - gt_root)).square().mean()
        total = sum(float(self.loss_weights[k]) * terms[k] for k in terms)
        out = {"loss": total}
        for k, value in terms.items():
            out["loss_" + k] = value.detach()
        out["mpjpe"] = out_mpjpe.detach()
        if not self.training and gt_geometry is not None:
            out["gt_joints_cam"] = gt_geometry[1].detach()
        return out

    def forward(self, data):
        pred = self._predict(data)
        losses = self._loss(
            pred, data) if self.training or "gt_pose_body_aa" in data else None
        if self.training:
            return losses
        if losses is not None:
            pred.update(losses)
        return pred
