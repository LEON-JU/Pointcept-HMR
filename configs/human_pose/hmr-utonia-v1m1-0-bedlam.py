_base_ = ["../_base_/default_runtime.py"]

patch_size = 256
seed = 51941598
model = dict(
    type="UtoniaHumanPose",
    pretrained_path=__import__("os").environ.get(
        "UTONIA_CKPT",
        "/home/juyiang/data/RGBD_human_pose/RGBD_HMR/utonia.pth"),
    use_lora=False,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    require_smpl=True,
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,
        enc_mode=True,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(patch_size, patch_size, patch_size, patch_size, patch_size),
        enable_flash=True,
        rope_base=10,
        freeze_encoder=False,
    ),
    backbone_out_channels=576,
    ops=dict(point3d_rope_backend="cuda",
             serialization_backend="cuda",
             pointnet2_backend="cuda"),
    max_point_tokens=2048,
    hidden_dim=512,
    num_heads=8,
    loss=dict(joint_2d=10.0,
              trans=5.0,
              pose=0.5,
              beta=0.001,
              cr_vert=1.0,
              smpl_joint=1.0),
    smpl=dict(
        models_dir=__import__("os").environ.get(
            "SMPLX_MODELS_DIR", "/home/juyiang/data/SMPL_models/models"),
        verts437_path=__import__("os").environ.get(
            "SMPLX_VERTS437_PATH",
            __import__("os").path.abspath(
                __import__("os").path.join(
                    "{{ fileDirname }}",
                    "../../assets/human_pose/smplx_verts437.pt"))),
    ),
)

data = dict(
    train=dict(type="BedlamHumanPoseDataset",
               pc_root=__import__("os").environ.get(
                   "BEDLAM_PC_ROOT", "/home/juyiang/data/bedlam/pc_row_shard"),
               gt_root=__import__("os").environ.get(
                   "BEDLAM_GT_ROOT", "/home/juyiang/data/bedlam/gt_shard"),
               smpl_models_dir=__import__("os").environ.get(
                   "SMPLX_MODELS_DIR", "/home/juyiang/data/SMPL_models/models"),
               training=True,
               human_pose_collate=True,
               filter_extreme_outliers=True,
               outlier_radius_quantile=0.95,
               outlier_radius_scale=1.0,
               outlier_min_radius_m=1.0,
               outlier_max_radius_m=1.0,
               outlier_min_keep_points=256,
               loop=1,
               augmentation=dict(flip_probability=0.5,
                                 rotation_axis_probability=0.5,
                                 rotation_center="root",
                                 jitter_probability=1.0,
                                 jitter_sigma=0.01,
                                 jitter_clip=0.05)),
    val=dict(
        type="BehaveHumanPoseDataset",
        manifest=__import__("os").environ.get(
            "BEHAVE_MANIFEST",
            "/home/juyiang/data/PointTransformerV3_HumanPointCloud/data/behave/behave_evaluation_full_shard"
        ),
        smpl_models_dir=__import__("os").environ.get(
            "SMPLX_MODELS_DIR", "/home/juyiang/data/SMPL_models/models"),
        training=False,
        human_pose_collate=True,
        filter_extreme_outliers=True,
        outlier_radius_quantile=0.95,
        outlier_radius_scale=2.0,
        outlier_min_radius_m=2.0,
        outlier_max_radius_m=2.0,
        outlier_min_keep_points=256),
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.02)
scheduler = dict(type="OneCycleLR",
                 max_lr=1e-4,
                 pct_start=0.05,
                 anneal_strategy="cos",
                 div_factor=10.0,
                 final_div_factor=10.0)
# Only backbone Blocks use 1/10 of the main LR; embedding/pooling keep the main LR.
param_dicts = [dict(keyword="backbone", module_type="Block", lr_scale=0.1)]
clip_grad = 1.0
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="HumanPoseEvaluator"),
    dict(type="CheckpointSaver",
         save_freq=None,
         save_last_freq=1000,
         save_epoch_freq=1,
         metric_mode="min")
]
batch_size = 256
batch_size_val = 256
# Total passes over the training dataset. Change this value for longer runs.
epoch = 2
# Evaluate every N optimizer steps, plus epoch end; 0 = epoch end only.
eval_every_step = 5000
enable_amp = True
amp_dtype = "bfloat16"
save_path = "exp/human_pose/hmr-utonia-v1m1-0-bedlam"
