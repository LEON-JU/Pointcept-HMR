_base_ = ["../_base_/default_runtime.py"]

model = dict(
    type="UtoniaHumanPose",
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,
        enc_mode=True,
        enc_depths=(1, ),
        enc_channels=(32, ),
        enc_num_head=(2, ),
        enc_patch_size=(16, ),
        stride=(),
        enable_flash=False,
        rope_base=None,
        order=("z", ),
        shuffle_orders=False,
    ),
    backbone_out_channels=32,
    ops=dict(point3d_rope_backend="torch",
             serialization_backend="torch",
             pointnet2_backend="torch"),
    max_point_tokens=8,
    hidden_dim=32,
    num_heads=4,
)
data = dict(
    train=dict(type="SyntheticHumanPoseDataset",
               length=2,
               points=64,
               human_pose_collate=True),
    val=dict(type="SyntheticHumanPoseDataset",
             length=1,
             points=64,
             human_pose_collate=True),
)
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.02)
scheduler = dict(type="OneCycleLR", max_lr=1e-4)
hooks = [
    dict(type="IterationTimer", warmup_iter=0),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None)
]
batch_size = 2
batch_size_val = 1
num_worker = 0
epoch = 1
eval_every_step = 0
evaluate = False
enable_amp = False
enable_swanlab = False
save_path = "/tmp/pointcept-human-pose-smoke"
