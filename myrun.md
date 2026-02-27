# π₀.₅ 模型微调训练流程

## 1. 环境配置

### 1.1 激活虚拟环境
```bash
source activate pi5
```

### 1.2 设置 HuggingFace 镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 1.3 安装依赖
```bash
# 进入项目目录
cd /data3/yinmenghao/code/openpi

# 安装依赖
uv sync
uv pip install -e .

# 应用 transformers 补丁
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

## 2. 数据准备

### 2.1 数据转换
使用整合后的转换脚本将自定义数据集转换为 LeRobot v2 格式：

```bash
# 执行数据转换脚本
python convert_assembly_bun_with_stats.py \
    --raw-data-dir /data3/yinmenghao/code/raw_data/assembly_bun \
    --output-dir /data3/yinmenghao/code/openpi/data \
    --split-ratio 0.9 \
    --task-description "move the pineapple bun from the conveyor belt to the center of the plate" \
    --fps 10
```

转换后的数据集结构：
```
data/
├── assembly_bun_train/
│   ├── data/
│   │   └── chunk-000/
│   │       └── *.parquet
│   ├── meta/
│   │   ├── episodes.jsonl
│   │   ├── info.json
│   │   ├── modality.json
│   │   ├── relative_stats.json
│   │   ├── stats.json
│   │   └── tasks.jsonl
│   └── videos/
│       └── chunk-000/
│           ├── image/
│           └── wrist_image/
└── assembly_bun_val/
    ├── data/
    ├── meta/
    └── videos/
```

## 3. 数据归一化统计计算

在开始训练前，需要计算数据集的归一化统计信息。这一步非常重要，否则训练会失败。

```bash
# 计算 LoRA 配置的归一化统计
uv run python scripts/compute_norm_stats.py --config-name=pi05_assembly_bun_lora

# 计算全参数配置的归一化统计
uv run python scripts/compute_norm_stats.py --config-name=pi05_assembly_bun_full
```

## 4. 训练配置

### 4.1 基于LoRA的微调配置 (JAX)
训练配置位于 `src/openpi/training/config.py` 文件中，已经配置了基于LoRA的 `pi05_assembly_bun_lora` 训练配置，适用于单GPU训练：

```python
TrainConfig(
    name="pi05_assembly_bun_lora",
    model=pi0_config.Pi0Config(
        pi05=True, 
        action_horizon=10, 
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",  # 启用 Paligemma 的 LoRA
        action_expert_variant="gemma_300m_lora",  # 启用 action expert 的 LoRA
    ),
    data=LeRobotAssemblyBunDataConfig(
        repo_id="assembly_bun_train",
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=128,  # 批次大小，适合单GPU LoRA训练
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1000,  # 预热步数
        peak_lr=5e-5,  # 学习率
        decay_steps=20000,  # 与训练步数匹配
        decay_lr=5e-6,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,  # JAX支持EMA
    # 使用JAX权重加载器
    weight_loader=weight_loaders.CheckpointWeightLoader("/data3/yinmenghao/code/openpi/local_model/pi05_base_lora/params"),
    num_train_steps=20000,  # 设置为20000训练步数
    fsdp_devices=1,  # 使用1个GPU进行训练
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),  # 冻结非 LoRA 参数，只训练 LoRA 权重
)
```

### 4.2 全参数微调配置 (PyTorch)
同时还配置了全参数微调的 `pi05_assembly_bun_full` 训练配置，适用于6GPU训练：

```python
TrainConfig(
    name="pi05_assembly_bun_full",
    model=pi0_config.Pi0Config(
        pi05=True, 
        action_horizon=10, 
        discrete_state_input=False,
        paligemma_variant="gemma_2b",  # 使用常规变体进行全参量训练
        action_expert_variant="gemma_300m",  # 使用常规变体进行全参量训练
    ),
    data=LeRobotAssemblyBunDataConfig(
        repo_id="assembly_bun_train",
        base_config=DataConfig(prompt_from_task=True),
    ),
    batch_size=128,  # 批次大小，适合PyTorch全参量训练
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1000,  # 预热步数
        peak_lr=5e-5,  # 学习率
        decay_steps=20000,  # 与训练步数匹配
        decay_lr=5e-6,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=None,  # PyTorch 不支持 EMA
    # 使用PyTorch权重路径
    pytorch_weight_path="/data3/yinmenghao/code/openpi/local_model/pi05_base_full",
    num_train_steps=20000,  # 设置为20000训练步数
    fsdp_devices=6,  # 使用6个GPU进行训练
    freeze_filter=None,  # 全参量训练，不需要冻结过滤器
)
```

### 4.3 配置说明
- `model`: 模型配置
  - `paligemma_variant`: 设置为 `gemma_2b_lora` 启用 Paligemma 的 LoRA，或 `gemma_2b` 用于全参数训练
  - `action_expert_variant`: 设置为 `gemma_300m_lora` 启用 action expert 的 LoRA，或 `gemma_300m` 用于全参数训练
- `data`: 数据配置，使用 `LeRobotAssemblyBunDataConfig` 处理自定义数据集
- `batch_size`: 批次大小，LoRA 训练适合单GPU训练，全参数训练适合多GPU
- `lr_schedule`: 学习率调度，调整为适合20000步训练的参数
- `ema_decay`: JAX 支持 EMA，PyTorch 不支持
- `weight_loader`: JAX 权重加载器路径（仅 LoRA 配置）
- `pytorch_weight_path`: PyTorch 权重路径（仅全参数配置）
- `num_train_steps`: 训练步数，设置为20000
- `fsdp_devices`: GPU 数量，LoRA 配置使用1个GPU，全参数配置使用6个GPU
- `freeze_filter`: LoRA 配置用于冻结非 LoRA 参数

## 5. 执行训练

### 5.1 LoRA 微调 (单GPU, JAX)
使用单个GPU进行LoRA微调：

**注意**：已成功下载完整的JAX模型权重，可直接执行以下命令开始训练。

```bash
# 执行 LoRA 训练（使用GPU 0）
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py pi05_assembly_bun_lora --exp_name assembly_bun_lora_finetune --save_interval 2000 --wandb-enabled --overwrite
```

#### 训练输出示例
训练开始后，会打印模型参数信息和可训练参数数量：

```
Total parameters: 2.05B
Trainable parameters: 4.82M  # 仅LoRA参数可训练，其他参数冻结
```

训练过程中会打印每步的指标：

```
Step 0: grad_norm=0.6004, loss=0.1473, param_norm=1803.7705
```

### 5.2 全参数微调 (6GPU, PyTorch)
使用6个GPU进行全参数微调：

```bash
# 执行全参数训练（使用6张GPU，设备ID：0,1,2,3,4,5）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 uv run torchrun --standalone --nnodes=1 --nproc_per_node=6 scripts/train_pytorch.py pi05_assembly_bun_full --exp_name assembly_bun_full_finetune --save_interval 1000 --wandb-enabled --overwrite
```

#### 训练输出示例
训练开始后，会打印可训练参数数量：

```
Trainable parameters: 2.05B  # 所有参数均可训练
```

训练过程中会打印每步的指标，与LoRA训练类似。

### 5.3 单GPU全参数微调测试 (1GPU, PyTorch)
使用单个GPU进行全参数微调测试（适合快速验证）：

```bash
# 执行单GPU全参数训练测试
CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/train_pytorch.py pi05_assembly_bun_full --exp_name assembly_bun_full_test --save_interval 10 --num_train_steps 10 --wandb-enabled --overwrite
```

### 5.4 训练参数说明
- `--exp_name`: 实验名称，用于区分不同训练运行
- `--save_interval`: 模型保存间隔（步数），小数据集可以设置更小的值
- `--overwrite`: 覆盖现有实验目录
- `--wandb-enabled`: 启用 Weights & Biases 日志记录
- `--num_train_steps`: 训练步数（可覆盖配置中的设置）

### 5.5 LoRA 微调优势
- **低显存占用**: 只训练少量 LoRA 参数，显存占用大幅降低
- **更快的训练速度**: 训练参数减少，训练速度加快
- **适合小数据集**: 只调整模型的一小部分参数，避免过拟合
- **便于模型合并**: 训练完成后可以将 LoRA 权重合并到基础模型中

## 6. 监控训练

### 6.1 Weights & Biases 监控
训练过程中，可通过 Weights & Biases 监控训练进度和指标：
- 登录 Weights & Biases 账号
- 在项目页面查看训练指标：损失函数、学习率、梯度范数等

### 6.2 日志文件
训练日志保存在实验目录中：
```
# LoRA 训练日志
checkpoints/pi05_assembly_bun_lora/assembly_bun_lora_finetune/
├── logs/
│   └── train.log
└── ...

# 全参数训练日志
checkpoints/pi05_assembly_bun_full/assembly_bun_full_finetune/
├── logs/
│   └── train.log
└── ...
```

## 7. 模型使用

### 7.1 加载训练好的模型

#### 加载 LoRA 训练的模型
```python
from openpi.training import config as _config
from openpi.policies import policy_config

# 加载 LoRA 配置
config = _config.get_config("pi05_assembly_bun_lora")

# 加载训练好的模型
checkpoint_dir = "/data3/yinmenghao/code/openpi/checkpoints/pi05_assembly_bun_lora/assembly_bun_lora_finetune/20000"  # 替换为实际训练步数
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 执行推理
example = {
    "observation/state": [0.5, -0.1, 0.4, 0.9, -0.4, 0.0, 0.0, 0.7],  # 示例状态
    "observation/image": None,  # 实际使用时提供图像
    "observation/wrist_image": None,  # 实际使用时提供手腕图像
    "prompt": "move the pineapple bun from the conveyor belt to the center of the plate"
}

action_chunk = policy.infer(example)["actions"]
print(f"Predicted actions: {action_chunk}")
```

#### 加载全参数训练的模型
```python
from openpi.training import config as _config
from openpi.policies import policy_config

# 加载全参数配置
config = _config.get_config("pi05_assembly_bun_full")

# 加载训练好的模型
checkpoint_dir = "/data3/yinmenghao/code/openpi/checkpoints/pi05_assembly_bun_full/assembly_bun_full_finetune/20000"  # 替换为实际训练步数
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 执行推理
example = {
    "observation/state": [0.5, -0.1, 0.4, 0.9, -0.4, 0.0, 0.0, 0.7],  # 示例状态
    "observation/image": None,  # 实际使用时提供图像
    "observation/wrist_image": None,  # 实际使用时提供手腕图像
    "prompt": "move the pineapple bun from the conveyor belt to the center of the plate"
}

action_chunk = policy.infer(example)["actions"]
print(f"Predicted actions: {action_chunk}")
```

### 7.2 启动策略服务器

#### 启动 LoRA 训练模型的策略服务器
```bash
# 启动策略服务器
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_assembly_bun_lora \
    --policy.dir=/data3/yinmenghao/code/openpi/checkpoints/pi05_assembly_bun_lora/assembly_bun_lora_finetune/20000
```

#### 启动全参数训练模型的策略服务器
```bash
# 启动策略服务器
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_assembly_bun_full \
    --policy.dir=/data3/yinmenghao/code/openpi/checkpoints/pi05_assembly_bun_full/assembly_bun_full_finetune/20000
```

## 8. 常见问题解决

### 8.1 GPU 显存不足
- 减少 `batch_size`
- 降低 `fsdp_devices` 数量
- 关闭 EMA：`ema_decay=None`（仅 JAX）

### 8.2 训练过程中出现 NaN
- 降低学习率
- 检查数据归一化
- 增加梯度裁剪阈值

### 8.3 模型加载失败
- 检查 `weight_loader` 或 `pytorch_weight_path` 是否指向正确的模型目录
- 确保转换后的模型文件完整

### 8.4 归一化统计错误
- 确保在训练前运行了 `scripts/compute_norm_stats.py`
- 检查配置名称是否正确

### 8.5 JAX LoRA 训练错误
- 确保已获得完整的 JAX 模型权重
- 检查 `weight_loader` 路径是否正确

## 9. 训练流程总结

1. **环境配置**：激活虚拟环境，设置镜像，安装依赖
2. **数据准备**：使用转换脚本将自定义数据转换为 LeRobot v2 格式
3. **计算归一化统计**：运行 `scripts/compute_norm_stats.py` 生成归一化统计
4. **训练配置**：根据需求选择 LoRA 或全参数配置
5. **执行训练**：使用相应的脚本启动训练
   - LoRA 训练：`scripts/train.py pi05_assembly_bun_lora`
   - 全参数训练：`scripts/train_pytorch.py pi05_assembly_bun_full`
6. **监控训练**：通过 Weights & Biases 或日志文件监控训练进度
7. **模型使用**：加载训练好的模型进行推理或部署策略服务器

按照以上流程，即可完成 π₀.₅ 模型在自定义数据集上的微调训练。

# JAX模型推理流程

本文档总结了使用微调后的JAX模型进行推理的完整流程，包括策略服务器启动、推理脚本运行和结果可视化。

## 1. 启动策略服务器

使用微调后的JAX模型启动策略服务器：

```bash
cd /data3/yinmenghao/code/openpi
source .venv/bin/activate
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_assembly_bun_lora --policy.dir=checkpoints/pi05_assembly_bun_lora/assembly_bun_lora_finetune_0110/19999
```

服务器将在8000端口启动，等待推理请求。

## 2. 运行推理脚本

使用优化后的推理脚本对验证集进行推理：

```bash
cd /data3/yinmenghao/code/openpi/inference_with_jax
source ../.venv/bin/activate
python infer_assembly_bun.py
```

### 2.1 推理脚本参数

推理脚本支持以下命令行参数：

```bash
python infer_assembly_bun.py --checkpoint-dir <checkpoint_path> --val-data-dir <val_data_path> --ws-url <websocket_url>
```

- `--checkpoint-dir`：checkpoint目录路径，默认为`/data3/yinmenghao/code/openpi/checkpoints/pi05_assembly_bun_lora/assembly_bun_lora_finetune_0110/19999`
- `--val-data-dir`：验证数据目录路径，默认为`/data3/yinmenghao/code/openpi/data/assembly_bun_val/data/chunk-000`
- `--ws-url`：策略服务器的WebSocket URL，默认为`ws://localhost:8000`

### 2.2 推理结果保存

推理脚本将：
- 从checkpoint路径中提取信息，生成简化的checkpoint标识
- 生成日期时间字符串（精确到分钟）
- 构建包含checkpoint信息和日期时间的输出目录，格式为：
  ```
  openpi/inference_results/<checkpoint_id>_date_<YYYYMMDD_HHMM>
  ```
- 对验证集中的所有episode进行推理
- 将推理结果保存到该输出目录下

## 3. 分析推理结果

使用分析脚本查看推理结果的统计信息：

```bash
cd /data3/yinmenghao/code/openpi/inference_with_jax
source ../.venv/bin/activate
python analyze_inference_results.py
```

**注意**：分析脚本需要修改以支持指定推理结果目录。目前它默认分析`inference_results/episode_000000_inference_results.json`文件。

## 4. 可视化关节状态

### 4.1 可视化所有episode

使用优化后的可视化脚本生成所有episode的关节状态对比曲线图：

```bash
cd /data3/yinmenghao/code/openpi/inference_with_jax
source ../.venv/bin/activate
python visualize_joint_states.py --inference-results-dir <inference_results_dir>
```

其中`<inference_results_dir>`是推理脚本生成的输出目录，例如：

```bash
python visualize_joint_states.py --inference-results-dir /data3/yinmenghao/code/openpi/inference_results/pi05_assembly_bun_lora_19999_date_20260112_1646
```

### 4.2 可视化单个episode

如果只需要可视化特定的episode：

```bash
python visualize_joint_states.py --inference-results-dir <inference_results_dir> --episode-idx 0
```

### 4.3 可视化脚本参数

可视化脚本支持以下命令行参数：

- `--inference-results-dir`：推理结果目录路径（必填）
- `--val-data-dir`：验证数据目录路径，默认为`/data3/yinmenghao/code/openpi/data/assembly_bun_val/data/chunk-000`
- `--episode-idx`：要可视化的episode索引（可选，默认可视化所有episode）

### 4.4 可视化结果保存

可视化脚本将：
- 为每个关节生成对比曲线
- 蓝色曲线表示真实关节状态
- 红色虚线表示预测关节状态
- 将图像保存到推理结果目录下，与推理结果文件放在一起

## 5. 完整的推理和可视化流程

### 5.1 步骤1：启动策略服务器

```bash
cd /data3/yinmenghao/code/openpi
source .venv/bin/activate
python scripts/serve_policy.py policy:checkpoint --policy.config=pi05_assembly_bun_lora --policy.dir=checkpoints/pi05_assembly_bun_lora/assembly_bun_lora_finetune_0110/19999
```

### 5.2 步骤2：运行推理脚本

```bash
cd /data3/yinmenghao/code/openpi/inference_with_jax
source ../.venv/bin/activate
python infer_assembly_bun.py
```

### 5.3 步骤3：获取推理结果目录

推理脚本运行完成后，会输出推理结果目录路径，例如：

```
Inference completed. Results saved to: /data3/yinmenghao/code/openpi/inference_results/pi05_assembly_bun_lora_19999_date_20260112_1646
```

### 5.4 步骤4：运行可视化脚本

使用步骤3中获得的推理结果目录路径，运行可视化脚本：

```bash
python visualize_joint_states.py --inference-results-dir /data3/yinmenghao/code/openpi/inference_results/pi05_assembly_bun_lora_19999_date_20260112_1646
```

## 6. 推理结果文件说明

优化后的脚本将所有结果保存在同一个目录下：

```
<inference_results_dir>/
├── all_inference_results.json       # 所有episode的推理结果
├── episode_000000_inference_results.json  # 第0个episode的推理结果
├── episode_000000_joint_states_comparison.png  # 第0个episode的关节状态对比图
├── episode_000001_inference_results.json  # 第1个episode的推理结果
├── episode_000001_joint_states_comparison.png  # 第1个episode的关节状态对比图
└── ...  # 其他episode的结果文件
```

## 7. 重要文件路径

- 微调模型路径：`checkpoints/pi05_assembly_bun_lora/assembly_bun_lora_finetune_0110/19999`
- 验证集路径：`data/assembly_bun_val`
- 推理脚本目录：`inference_with_jax/`
- 推理结果目录：`inference_results/`

## 8. 注意事项

1. 确保策略服务器正在运行，然后再运行推理脚本
2. 推理脚本和可视化脚本需要在虚拟环境中运行
3. 可视化脚本需要安装matplotlib库
4. 可以根据需要修改脚本中的参数，如checkpoint路径、验证数据路径等
5. 推理结果目录包含了完整的推理结果和可视化图像，便于管理和比较不同checkpoint的推理效果
