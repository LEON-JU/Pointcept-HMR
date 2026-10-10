# 模型结构

正式模型由 `configs/human_pose/hmr-utonia-v1m1-0-bedlam.py` 构建，入口为 `pointcept/models/human_pose/utonia_hmr.py::UtoniaHumanPose`。主干参数回归结构沿用 RGBD_HMR 的设计，并使用 Pointcept 的 PointTransformerV3/Utonia、registry 和 trainer。

```text
packed point cloud: coord + rgb + normal (9 channels)
        |
        v
Utonia PT-v3m3 encoder, 5 stages, output 576 channels
        |                                  |
        |                                  +--> PointNet++ MSG --> global feature (B, 1024)
        |
        +--> FPS 2048 deep points --> xyz + projected feature = memory (B, 2048, 512)
                                                 |
                         22 joint queries + 1 root query
                                                 |
                                   2-layer Transformer decoder
                            pose6d (B,22,6), trans_delta (B,3), betas (B,10)
                                                 |
                               differentiable SMPL-X (when assets configured)
```

## Utonia encoder

正式配置使用 9 通道输入 `[xyz, rgb, normal]`，从 `UTONIA_CKPT` 加载权重，默认路径为 `/home/juyiang/data/RGBD_human_pose/RGBD_HMR/utonia.pth`。encoder 配置如下：

| 参数 | 正式配置 |
|---|---|
| 类型 | `PT-v3m3`, `enc_mode=True` |
| 深度 | `(3, 3, 3, 12, 3)` |
| 通道 | `(54, 108, 216, 432, 576)` |
| heads | `(3, 6, 12, 24, 32)` |
| patch size | `(32, 32, 32, 32, 32)` |
| stride | `(2, 2, 2, 2)` |
| serialization orders | `z`, `z-trans`, `hilbert`, `hilbert-trans` |
| PointRoPE base | `10` |
| Flash attention | 开启 |

模型会解开 PT-v3 pooling trace，使用最深层点的 `coord/feat/offset`。输出 feature channel 为 576。输入 voxel size 由 dataset `grid_size=0.01` 控制。

## 全局特征与点 token

`PointNetPPGlobalFusion` 对每个样本 FPS 取 64 个 center，使用三个 MSG 分支：半径比例 `(0.08, 0.16, 0.32)`、邻域点数 `(16, 32, 64)`。半径乘以样本点云 bounding-box 对角线。各分支特征拼成 384 维，再经全局 MLP 和 max pooling 得到 1024 维 global feature。

另一路对 encoder 深层点做 FPS，最多保留 2048 个 token。576 维点特征经 LayerNorm 和线性层投影到 509 维，并与 xyz 拼成 512 维 memory。每个样本少于 2048 点时，attention 使用 `key_padding_mask` 屏蔽 padding token。

## Query decoder 和回归头

- Query 包含 22 个关节 query 和 1 个可学习 root query；22 个关节 query 由 1024 维 global feature 经 `1024 -> 22*64 -> 512` 生成。
- 两层 decoder，每层含 8-head self-attention、对 point memory 的 cross-attention、512->1024->512 FFN 和残差 LayerNorm。
- `pose_head` 对 22 个关节输出 6D rotation。
- `trans_head` 将关节 query 降维后与 root query 融合，回归相对点云质心的 `trans_delta`。
- `beta_head` 对 22 个 query 做 max pooling，回归前 10 维 SMPL-X shape beta。

模型输出包括 `pred_pose6d (B,22,6)`、`pred_trans_delta (B,3)`、`pred_betas (B,10)`。配置 SMPL-X 后再输出 `pred_vertices` 和相机坐标系下 `pred_joints_cam (B,22,3)`。完整的绝对平移通过 `pred_trans_cam = pred_trans_delta + point_centroid` 还原。

SMPL-X 使用 neutral、10 betas、关闭 PCA、flat hand mean；预测手部姿态补零。可微预测分支参与 loss，GT 几何分支在 `no_grad` 下计算。
