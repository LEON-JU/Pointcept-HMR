# Utonia Human Pose in Pointcept

本文档记录 Pointcept 中 BEDLAM 点云人体参数回归任务的模型、数据、训练和运行依赖。目标是让后续维护从这里开始，不必先通读迁移代码。

## 文档导航

- [模型结构](01_model.md)：Utonia encoder、PointNet++ 融合、query decoder、SMPL-X 输出。
- [数据与损失](02_data_and_losses.md)：BEDLAM/BEHAVE shard、预处理、坐标系和监督项。
- [训练说明](03_training.md)：正式配置、启动命令、优化器、验证与 checkpoint。
- [算子与验证](04_ops_and_validation.md)：CUDA 扩展、构建、smoke test、性能记录和注意事项。

## 当前正式配置

- 配置：`configs/human_pose/hmr-utonia-v1m1-0-bedlam.py`
- 模型注册名：`UtoniaHumanPose`
- 训练集：BEDLAM `pc_row_shard` + `gt_shard`
- 验证集：BEHAVE packed shard
- 预训练 encoder：RGBD_HMR 项目的 `utonia.pth`
- 模型输出：22 个 body joints 的 pose 6D、相对点云质心的 transl、10 维 betas；配置 SMPL-X 资产后也输出预测 vertices/joints
- 正式训练默认：global batch 64、20 data passes、bf16、AdamW、OneCycleLR；默认在这 20 passes 后评估/保存

本机默认数据和模型资产路径在配置中通过环境变量覆盖。路径及变量清单见 [训练说明](03_training.md)。

`configs/human_pose/smoke.py` 和 `tools/benchmarks/human_pose_smoke.py` 只用于小型 CUDA 模型/数据契约测试；它不使用 BEDLAM、预训练 checkpoint 或正式 SMPL-X 监督，不能作为训练配置。

## 主要实现位置

| 范围 | 路径 |
|---|---|
| 回归模型与 SMPL-X loss | `pointcept/models/human_pose/utonia_hmr.py` |
| PointNet++ global fusion | `pointcept/models/human_pose/pointnet2_fusion.py` |
| BEDLAM、BEHAVE、collate | `pointcept/datasets/human_pose.py` |
| BEHAVE MPJPE evaluator | `pointcept/engines/hooks/human_pose_evaluator.py` |
| 正式配置 | `configs/human_pose/hmr-utonia-v1m1-0-bedlam.py` |
| CUDA 扩展源码 | `libs/human_pose_ops/{pointnet2,serialization,point3d_rope}/` |
| 扩展构建脚本 | `tools/build_human_pose_ops.sh` |
