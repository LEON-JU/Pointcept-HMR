# 训练说明

正式入口沿用 Pointcept 的 `tools/train.py` 和 `DefaultTrainer`，正式配置是 `configs/human_pose/hmr-utonia-v1m1-0-bedlam.py`。训练集、验证集、batch collate、优化器和 evaluator 均经 Pointcept registry 注册。

## 默认超参

| 参数 | 值 |
|---|---:|
| global batch | 256 |
| epoch | 2 |
| eval_every_step | `0`：只在 epoch 末尾评估；正数：每 N 个 optimizer step 评估，并在 epoch 末尾评估（重合只执行一次） |
| optimizer | AdamW, `lr=1e-4`, `weight_decay=0.02` |
| encoder block LR | `1e-5`，匹配 `backbone.enc.enc` 参数名 |
| scheduler | OneCycleLR |
| AMP | 开启，bfloat16 |
| gradient accumulation | 1 |
| gradient clip | 1.0 |
| DataLoader workers | 配置继承 Pointcept runtime 的总 `num_worker=16`，分配到各 GPU |

训练日志同时写入 TensorBoard 和 SwanLab。默认配置使用 `swanlab_project="pointcept"`、`swanlab_mode="online"`；可将 `enable_swanlab=False` 关闭，或改为 `local`/`offline` 模式。SwanLab 本地日志位于 `save_path/swanlab`。

batch size 是所有 GPU 的总 batch；4 GPU 时每卡 batch 64。训练采用 DistributedSampler；评估用不重复样本的 `HumanPoseDistributedEvalSampler`。

## 本机运行

首先确认 BEDLAM PC/GT shard、Utonia checkpoint、SMPL-X 模型目录以及 437 顶点索引文件可读。三类 CUDA 扩展需要与当前 PyTorch/CUDA/Python ABI 匹配：

```bash
cd /home/juyiang/data/RGBD_human_pose/Pointcept
export PYTHONPATH=.
export UTONIA_CKPT=/home/juyiang/data/RGBD_human_pose/RGBD_HMR/utonia.pth
export BEDLAM_PC_ROOT=/home/juyiang/data/bedlam/pc_row_shard
export BEDLAM_GT_ROOT=/home/juyiang/data/bedlam/gt_shard
export BEHAVE_MANIFEST=/home/juyiang/data/PointTransformerV3_HumanPointCloud/data/behave/behave_evaluation_full_shard
export SMPLX_MODELS_DIR=/home/juyiang/data/SMPL_models/models
bash tools/build_human_pose_ops.sh
python tools/train.py --config-file configs/human_pose/hmr-utonia-v1m1-0-bedlam.py --num-gpus 4 --options save_path=exp/human_pose/hmr-utonia-v1m1-0-bedlam
```

若扩展已构建且二进制仍匹配当前环境，可跳过构建步骤。单卡调试可把 `--num-gpus` 设为 1，并用 `--options batch_size=16 epoch=1 eval_every_step=0`；正式配置 `require_smpl=True`，仍需真实 SMPL-X 资产。路径也可以直接作为配置 overrides 传入，例如 `data.train.pc_root=...`、`data.train.gt_root=...`、`data.val.manifest=...`、`model.pretrained_path=...`、`model.smpl.models_dir=...`。

运行前会在模型初始化时加载 Utonia checkpoint，并严格检查 backbone 除 `embedding.mask_token` 外没有缺失或额外参数。训练产物写入 `save_path`，包含备份配置、日志、tensorboard 和 checkpoint。当前配置每 1000 个 optimizer step 原子更新一次 `model_last.pth`，每个 global epoch 保存一份带训练状态的 `epoch_N.pth`；BEHAVE 上 `smpl_joint + cr_vert` 最低的权重另存为 `model_best.pth`。MPJPE 仍作为验证日志指标。

中断恢复时将 `resume=True` 并把 `weight` 指向 `model_last.pth` 或某个 `epoch_N.pth`。step checkpoint 保存当前训练周期、已消费 batch 数、global step、模型、optimizer、scheduler、AMP scaler 和 best metric；恢复时会在当前周期跳过已消费的 batch。恢复不保存 DataLoader worker 的随机数状态，因此数据增强随机序列不会逐样本完全复现。

## 单步训练

Pointcept `DefaultTrainer.run_step` 将 batch tensor 异步搬到 CUDA，在 bf16 autocast 下调用模型。模型训练态返回包含 `loss` 和各项监控量的字典。trainer 对 loss 做梯度累积缩放，反向后 unscale、clip grad norm、optimizer step、更新 scaler 和 scheduler。`IterationTimer`/`InformationWriter` 负责耗时和标量日志。

## 配置维护约定

- 正式实验从 `hmr-utonia-v1m1-0-bedlam.py` 复制/覆盖，不要把 smoke 模型维度用于正式权重。
- 变更输入通道、encoder stage、patch size、token 数或 SMPL-X 参数时，记录配置与 checkpoint 的对应关系；预训练 checkpoint 是 Utonia encoder，不是完整 HMR 回归头。
- `epoch` 直接控制总训练轮数，当前为 `2`，每轮训练集只遍历一次。`eval_every_step=0` 默认仅在轮末评估；例如设为 `5000`，则在全局 optimizer step 5000、10000 等位置及轮末评估，重合时只评估一次。跨 epoch 不重置计数，梯度累积的 microbatch 和 AMP 跳过的更新不计为 optimizer step。命令行可独立设置 `--options epoch=2 eval_every_step=5000`。旧 `eval_epoch` 已移除；scheduler 总步数由总轮数和 loader 长度决定。
- 若更改 shard 数据格式，维持 manifest、sample index、shard row 及长度 sidecar 的一致性。
