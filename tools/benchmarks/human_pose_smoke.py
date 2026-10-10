"""Run a CPU registry/data/model contract smoke test with a tiny PT-v3 encoder."""
from __future__ import annotations

import argparse
import torch

import pointcept.models  # noqa: F401
import pointcept.datasets  # noqa: F401
from pointcept.models.builder import build_model
from pointcept.datasets.builder import build_dataset
from pointcept.datasets.human_pose import human_pose_collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=64)
    args = parser.parse_args()
    dataset = build_dataset(
        dict(type="SyntheticHumanPoseDataset", length=2, points=args.points))
    batch = human_pose_collate_fn([dataset[0], dataset[1]])
    model = build_model(
        dict(
            type="UtoniaHumanPose",
            backbone=dict(type="PT-v3m3",
                          in_channels=9,
                          enc_mode=True,
                          enc_depths=(1, ),
                          enc_channels=(32, ),
                          enc_num_head=(2, ),
                          enc_patch_size=(16, ),
                          stride=(),
                          enable_flash=False,
                          rope_base=None,
                          order=("z", )),
            backbone_out_channels=32,
            ops=dict(point3d_rope_backend="torch",
                     serialization_backend="torch",
                     pointnet2_backend="torch"),
            max_point_tokens=8,
            hidden_dim=32,
            num_heads=4,
        ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        raise RuntimeError(
            "PT-v3 smoke requires CUDA because spconv implicit GEMM is CUDA-only"
        )
    model = model.to(device)
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    model.train()
    prediction = model._predict(batch)
    output = model._loss(prediction, batch)
    assert prediction["pred_pose6d"].shape == (2, 22, 6)
    assert prediction["pred_trans_delta"].shape == (2, 3)
    assert prediction["pred_betas"].shape == (2, 10)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    print(
        "human pose smoke passed", {
            k: tuple(v.shape)
            for k, v in output.items()
            if isinstance(v, torch.Tensor) and v.ndim
        })


if __name__ == "__main__":
    main()
