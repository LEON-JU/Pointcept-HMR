# 算子与验证

正式配置把三类热点算子设为 CUDA 后端：`point3d_rope_backend="cuda"`、`serialization_backend="cuda"`、`pointnet2_backend="cuda"`。扩展源码分别位于 `libs/human_pose_ops/point3d_rope`、`libs/human_pose_ops/serialization` 和 `libs/human_pose_ops/pointnet2`。

## 算子用途

| 扩展 | 用途 |
|---|---|
| Point3D RoPE | 替换 Utonia attention 中的连续 xyz PointRoPE；保留自定义 autograd backward |
| serialization | 为 Pointcept PT-v3 提供 CUDA Morton/Z-order 和 Hilbert 编码；由 `POINTCEPT_SERIALIZATION_BACKEND=cuda` 选择 |
| PointNet++ | 加速 FPS、ball query、gather/group；用于 global fusion 和 point token 采样 |

`UtoniaHumanPose._configure_ops` 会检查请求的扩展是否可用。RoPE backend patch 在模型构建时施加于 backbone；序列化 backend 在模型构建时写入环境变量。若扩展不可用，CUDA 配置会直接报错，不会静默回退。Torch/reference 实现主要供小型 smoke/debug 使用，PointNet++ 的逐点 reference FPS 不适合正式大点云训练。

构建命令：

```bash
cd /home/juyiang/data/RGBD_human_pose/Pointcept
PYTHON_BIN=/path/to/active/env/bin/python bash tools/build_human_pose_ops.sh
```

建议在准备训练的同一 Python、PyTorch、CUDA toolkit 和编译器环境中构建。脚本依次编译三个扩展到 repo 根路径；升级环境或清理扩展二进制后需重新构建。

## 检查与 smoke test

```bash
cd /home/juyiang/data/RGBD_human_pose/Pointcept
PYTHONPATH=. python tools/benchmarks/human_pose_smoke.py
```

smoke test 使用小型随机 PT-v3、随机合成点云，检查 registry、batch collate、模型输出形状、有限 loss 和 backward。它只检查 CUDA PT-v3 契约；不检查真实 BEDLAM shard、预训练 checkpoint、正式尺寸网络、真实 SMPL-X assets、验证集或性能。

正式运行前可分别检查：

1. 三个扩展均可 import，且 build info 中 Torch/CUDA 版本与当前环境相符。
2. PC/GT `manifest.json`、sample index 和 PC `length_rel_path` sidecars 存在。
3. `UTONIA_CKPT` 可加载且与配置的 PT-v3 encoder 兼容。
4. `SMPLX_MODELS_DIR` 和 `assets/human_pose/smplx_verts437.pt` 存在。
5. 单 batch 前向/反向正常，训练 loss 有限，BEHAVE evaluator 能输出 MPJPE。

## 一次性耗时记录

2026-10-09 在本机 RTX 4090、单卡、batch 16、bf16、CUDA 三算子后端，跑正式 RGBD_HMR 原项目 20-step 短训练；排除前 3 个 step 后 17-step 平均完整 step 约 **228.36 ms**，中位数 **233.04 ms**。原项目该次配置 `patch_size=32`、`max_point_tokens=2048`、全 5-stage encoder；数据预取等待约 0.02 ms，CPU 增强约 42.91 ms（与 GPU 计算重叠），主线程 batch prepare 约 2.56 ms。记录文件在 `/tmp/rgbdhmr_profile/compare_original_bs16_fixed_20261009T114833/logs/train_step_profile.csv`。

此前 Pointcept 侧记录约 189.5 ms/step，但两个结果不构成严格同配置 benchmark：此前记录对应的 Pointcept 正式 config 使用 encoder `enc_patch_size=1024`，而当前正式人姿配置已改为 `enc_patch_size=32`，数据 pipeline 和 trainer timer 也不同。不要据此断言迁移后更快或更慢；要比较性能应在同一批样本、相同 local batch、点数过滤、模型 patch/token 配置和计时边界下重新测量。
