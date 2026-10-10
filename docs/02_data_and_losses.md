# 数据与损失

实现位于 `pointcept/datasets/human_pose.py` 和 `pointcept/models/human_pose/utonia_hmr.py`。点坐标和 transl 使用相机坐标系、米制单位。

## BEDLAM 训练数据

`BedlamHumanPoseDataset` 读取 PC shard 和 GT shard 两个根目录中的 `manifest.json` 及 sample index。默认路径：

- `BEDLAM_PC_ROOT=/home/juyiang/data/bedlam/pc_row_shard`
- `BEDLAM_GT_ROOT=/home/juyiang/data/bedlam/gt_shard`

每个 shard 是固定行宽的 NPY；sample index 把 scene/sequence/frame/person 映射到 shard row。PC manifest 中每个 shard 还需要 `length_rel_path` 指向的 point-count sidecar。Dataset 按 `min_points` 过滤样本，正式配置默认 128 点。每个 worker 使用 mmap，并对最近使用的两个 PC/GT shard 做 LRU 缓存。

每个 BEDLAM 样本读取点坐标、RGB、normal，以及 GT `pose_cam` 前 22 个 body joints、`betas` 前 10 维、`trans_cam`、`joints_cam` 和相机内参/图像尺寸。训练 batch 由 `human_pose_collate_fn` 拼接变长点并生成累计结束偏移 `offset`；GT 按 batch 维 stack。

## BEHAVE 验证数据

正式配置默认从 `BEHAVE_MANIFEST=/home/juyiang/data/PointTransformerV3_HumanPointCloud/data/behave/behave_evaluation_full_shard` 读取 BEHAVE shard。验证数据携带 GT SMPL-X 参数、关节、质心；没有相机内参，因此不计算 2D 投影项。`HumanPoseEvaluator` 汇总相机坐标系 22 关节 MPJPE，并在分布式时 all-reduce 误差总和与样本数。分布式验证 sampler 按 rank 切片，不补齐或重复样本。

## 预处理与增强

`_HumanPoseDataset._prepare` 执行：

1. 将 RGB 转为 float，若最大值大于 1 则除以 255；缺失 RGB/normal 时用零填充。
2. 训练时按原 RGBD-HMR 参数以 0.5 概率做 x 镜像；x/y/z 每个轴分别以 0.5 概率抽取 `[-pi, pi]` 随机旋转，默认绕 GT root 旋转（可设为绕点云质心），不绕相机原点；jitter 概率为 1.0，标准差 0.01、截断 0.05，仅抖动点坐标。
3. 计算点云质心并中心化坐标；`gt_trans_delta = gt_trans_cam - point_centroid`。
4. 按 1 cm voxel 生成 `grid_coord`，每个 voxel 训练时随机保留一个点，验证时确定性保留。
5. 构造 9 通道特征 `[centered xyz, rgb, normal]`。

旋转同时变换 coord、normal、joints、trans 和 root orientation，旋转中心与原 RGBD-HMR 的 GT-root 语义一致；镜像使用原项目的 22-joint 左右置换并同步变换 pose。数据增强在 dataset worker 中完成，Pointcept trainer 使用 pin-memory DataLoader。

## 损失

总损失由配置权重加权相加，默认权重如下：

| 项 | 权重 | 计算方式 |
|---|---:|---|
| `pose` | 0.5 | 预测 6D rotation 与 GT axis-angle 转 rotation 6D 的 MSE |
| `trans` | 5.0 | `pred_trans_delta` 与 `gt_trans_delta` 的 MSE |
| `beta` | 0.001 | 10 维 beta 的 MSE |
| `smpl_joint` | 1.0 | SMPL-X 前 22 关节先减 root，再与 GT root-relative joints 做 MSE |
| `cr_vert` | 1.0 | SMPL-X 预测和 GT 的 437 个采样顶点先减 root，再做 MSE |
| `joint_2d` | 10.0 | 相机内参投影到归一化图像坐标后做有效深度关节 MSE |

只有相机内参、图像尺寸和预测/GT joints 齐备时才启用 2D 项。损失实现返回 `loss` 以及 detach 的分项 `loss_*` 和 `mpjpe`，符合 Pointcept trainer 对模型输出字典的约定。BEHAVE 没有内参，2D loss 为零；其主要验证指标由 evaluator 计算。

437 顶点索引资产默认位于 `assets/human_pose/smplx_verts437.pt`。SMPL-X 模型目录需包含 `smplx` Python 包可读取的模型文件；默认路径 `/home/juyiang/data/SMPL_models/models`，可由 `SMPLX_MODELS_DIR` 覆盖。正式配置 `require_smpl=True`，缺少模型目录、索引资产或 `smplx` 包会在初始化时失败。
