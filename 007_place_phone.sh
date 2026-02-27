#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  Place Phone - Pi0.5 LoRA Fine-tuning Script
#  Task: Place phone onto target location
#  Model: Pi0.5 with LoRA (JAX, multi-GPU data parallel)
#  Action space: 7D [x, y, z, ax, ay, az, gripper] (axis-angle)
#
#  Usage:
#    bash 007_place_phone.sh              # 默认使用 GPU 0
#    bash 007_place_phone.sh 1            # 使用 GPU 1
#    bash 007_place_phone.sh 0,1,2,3      # 使用 GPU 0,1,2,3 (四卡并行)
#    bash 007_place_phone.sh 4,5,6,7      # 使用 GPU 4,5,6,7
# ══════════════════════════════════════════════════════════════

set -e  # Exit on error

# ──────────────────────── GPU 配置 ────────────────────────
# 从命令行参数读取 GPU ID，默认使用 GPU 0
GPU_IDS="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# 自动计算 GPU 数量（按逗号分隔计数）
NUM_GPUS=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')

# ──────────────────────── 环境配置 ────────────────────────
export HF_ENDPOINT=https://hf-mirror.com
# 设置 CUDA_HOME 指向当前的 Conda 环境根目录
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# 暂时清空并重新设置，确保 Conda 环境路径绝对优先
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
cd /home/yinmenghao/code/openpi

# ──────────────────────── 训练参数配置 ────────────────────────

# 配置名称（对应 config.py 中注册的训练配置）
CONFIG_NAME="pi05_place_phone_lora"

# 实验名称前缀（实际名称会自动递增版本号，避免覆盖已有 checkpoint）
EXP_NAME_PREFIX="place_phone_lora"

# 数据集 repo_id（对应 /data1/vla-data/processed/PI/data/ 下的目录名）
DATASET_REPO_ID="009_place_phone"

# 数据集路径
DATA_DIR="/data1/vla-data/processed/PI/data/${DATASET_REPO_ID}"

# ---- 单卡基准参数（以单卡为基准定义，多卡时自动缩放） ----
BASE_NUM_GPUS=1
BASE_BATCH_SIZE=16          # 单卡 batch size
BASE_NUM_TRAIN_STEPS=40000  # 单卡总训练步数
BASE_PEAK_LR="5e-5"        # 单卡峰值学习率

# ---- 多卡自动缩放 ----
# 全局 batch_size = 单卡 batch × GPU 数量（per-device batch 保持不变）
# 训练步数 = 单卡步数 / GPU 数量（保持总 epoch 数不变）
# 学习率 = 单卡 LR × sqrt(GPU 数量)（线性缩放太激进，sqrt 更稳定）
BATCH_SIZE=$((BASE_BATCH_SIZE * NUM_GPUS))
NUM_TRAIN_STEPS=$((BASE_NUM_TRAIN_STEPS / NUM_GPUS))

# 学习率 sqrt 缩放（用 python 计算浮点数）
PEAK_LR=$(python3 -c "import math; print(f'{${BASE_PEAK_LR} * math.sqrt(${NUM_GPUS}):.1e}')")
DECAY_LR=$(python3 -c "import math; print(f'{5e-6 * math.sqrt(${NUM_GPUS}):.1e}')")
WARMUP_STEPS=$((1000 / NUM_GPUS > 100 ? 1000 / NUM_GPUS : 100))

# ---- Checkpoint 配置 ----
# Checkpoint 输出根目录。最终路径: ${CHECKPOINT_DIR}/${CONFIG_NAME}/${EXP_NAME}/<step>/
CHECKPOINT_DIR="./checkpoints/place_phone_lora_0211"

# 自动递增实验版本号，避免覆盖已有 checkpoint
EXP_BASE_DIR="${CHECKPOINT_DIR}/${CONFIG_NAME}"
VERSION=1
while [ -d "${EXP_BASE_DIR}/${EXP_NAME_PREFIX}_v${VERSION}" ]; do
    VERSION=$((VERSION + 1))
done
EXP_NAME="${EXP_NAME_PREFIX}_v${VERSION}"
echo "  Auto-selected experiment name: ${EXP_NAME}"

# Checkpoint 保存间隔（固定 2000 步保存一次，不随 GPU 数缩放）
SAVE_INTERVAL=2000

# 最大保留 checkpoint 数量。超出时自动删除最旧的。
# 设为空字符串 "" 则保留所有。
MAX_CHECKPOINTS=5

# ---- Action Chunk Size (动作预测步数) ----
# 模型每次预测未来多少步的动作序列。
# 较大的 chunk 可以捕获更长期的动作规划，但需要更多数据来学习；
# 较小的 chunk 更容易学习，但只看到短期动作。
# 常用值: 10 (适合大多数任务), 16 (长距离任务), 50 (Pi0 默认)
# ⚠️ 注意: compute_norm_stats.py 读取 config.py 中硬编码的 action_horizon，
#    如果修改此值，需同步修改 config.py 中 pi05_place_phone_lora 配置的
#    action_horizon，并删除已缓存的 norm_stats.json 重新计算。
ACTION_HORIZON=10

# ---- DataLoader Worker 配置 ----
# 多卡 JAX 训练时，PyTorch DataLoader 使用 spawn 方式创建 worker 进程。
# 每个 worker 会初始化自己的 CUDA context，worker 过多会导致资源争抢、
# 进程被 kill（"DataLoader worker is killed by signal: Interrupt"）。
# 建议: 单卡可用 8-16，多卡建议 2-4 per GPU，最大不超过 CPU 核数的一半。
NUM_WORKERS=$((NUM_GPUS * 16 > 64 ? 64 : NUM_GPUS * 16))

# ---- 其他训练参数 ----
EMA_DECAY=0.999         # EMA 衰减系数（JAX 支持 EMA）
SEED=42
LOG_INTERVAL=100        # 每隔多少步打印一次训练指标

# ---- FSDP 配置 (模型并行) ----
# LoRA 参数量很小，不需要 FSDP 模型切分，纯数据并行即可。
# 仅当显存不足时才考虑增大 FSDP_DEVICES（必须能整除 NUM_GPUS）。
FSDP_DEVICES=1

# ---- W&B 日志 ----
WANDB_ENABLED=true      # 设为 false 可关闭 W&B 日志

# ---- 图像增强配置 (ColorJitter) ----
# 训练时随机扰动图像的亮度/对比度/饱和度/色相，提升模型对光照变化的泛化能力。
# 参数含义同 torchvision.transforms.ColorJitter / GR00T --color-jitter-params
# 设为 false 关闭图像增强（推理时自动不启用）
IMAGE_AUGMENT_ENABLED=true
IMAGE_AUGMENT_BRIGHTNESS=0.3    # 亮度: factor ∈ [1-b, 1+b]
IMAGE_AUGMENT_CONTRAST=0.4      # 对比度: factor ∈ [1-c, 1+c]
IMAGE_AUGMENT_SATURATION=0.5    # 饱和度: factor ∈ [1-s, 1+s]
IMAGE_AUGMENT_HUE=0.08          # 色相: shift ∈ [-h, h] × 360°

# ---- 动作扩散步数说明 ----
# Pi0 使用 Flow Matching（流匹配），不是传统的 DDPM 扩散模型。
# 训练时: 随机采样一个连续时间步 t ~ Beta(1.5, 1)，计算单步 flow matching loss，
#         不存在"扩散步数"这个概念，每个 batch 只需一次前向传播。
# 推理时: 使用迭代去噪，默认 num_steps=10（在 sample_actions 中设置）。
#         推理步数在启动策略服务器时配置，不在训练阶段设置。
#         减少步数(如5)可加速推理但降低质量，增加步数(如20)可提升质量但变慢。

# ──────────────────────── 打印配置摘要 ────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Multi-GPU Configuration Summary"
echo "══════════════════════════════════════════════════════════════"
echo "  GPU IDs:                ${GPU_IDS}"
echo "  Number of GPUs:         ${NUM_GPUS}"
echo "  Per-device batch size:  ${BASE_BATCH_SIZE}"
echo "  Global batch size:      ${BATCH_SIZE}"
echo "  Train steps:            ${NUM_TRAIN_STEPS} (base: ${BASE_NUM_TRAIN_STEPS} / ${NUM_GPUS} GPUs)"
echo "  Peak LR:                ${PEAK_LR} (base: ${BASE_PEAK_LR} × √${NUM_GPUS})"
echo "  Decay LR:               ${DECAY_LR}"
echo "  Warmup steps:           ${WARMUP_STEPS}"
echo "  Save interval:          ${SAVE_INTERVAL}"
echo "  Max checkpoints:        ${MAX_CHECKPOINTS:-unlimited}"
echo "  FSDP devices:           ${FSDP_DEVICES}"
echo "  Image augment:          ${IMAGE_AUGMENT_ENABLED} (B=${IMAGE_AUGMENT_BRIGHTNESS} C=${IMAGE_AUGMENT_CONTRAST} S=${IMAGE_AUGMENT_SATURATION} H=${IMAGE_AUGMENT_HUE})"
echo "══════════════════════════════════════════════════════════════"
echo ""

# ══════════════════════════════════════════════════════════════
#  Step 1: 计算归一化统计
# ══════════════════════════════════════════════════════════════

echo "============================================================"
echo "  Step 1: Computing normalization statistics..."
echo "============================================================"

# 检查 norm_stats 是否已经存在
NORM_STATS_PATH="./assets/${CONFIG_NAME}/${DATASET_REPO_ID}/norm_stats.json"
if [ -f "${NORM_STATS_PATH}" ]; then
    echo "  Norm stats already exist at: ${NORM_STATS_PATH}"
    echo "  Skipping computation. Delete the file to recompute."
else
    # 快速计算: 直接从 parquet 读取数值数据，跳过视频解码。
    # 逻辑与 compute_norm_stats.py 完全一致: action chunking + delta 转换 + RunningStats。
    uv run python -c "
import sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, 'src')
import openpi.shared.normalize as normalize

DATA_DIR = pathlib.Path('${DATA_DIR}')
ACTION_HORIZON = ${ACTION_HORIZON}
DELTA_MASK_DIMS = 6  # first 6 dims use delta, last dim (gripper) stays absolute

# Load all parquet files
parquet_files = sorted(DATA_DIR.glob('data/*/*.parquet'))
print(f'  Loading {len(parquet_files)} parquet files...')

state_stats = normalize.RunningStats()
action_stats = normalize.RunningStats()

for pf in parquet_files:
    df = pd.read_parquet(pf, columns=['state', 'actions', 'episode_index', 'frame_index'])
    states = np.array(df['state'].tolist(), dtype=np.float32)    # (N, 7)
    actions = np.array(df['actions'].tolist(), dtype=np.float32)  # (N, 7)
    n = len(states)

    # Build action chunks: for frame i, chunk = [actions[i], ..., actions[i+H-1]]
    for i in range(n - ACTION_HORIZON + 1):
        s = states[i]                                        # (7,)
        a_chunk = actions[i:i + ACTION_HORIZON].copy()       # (H, 7)
        # Delta conversion: a_chunk[:, :6] -= s[:6]
        a_chunk[:, :DELTA_MASK_DIMS] -= s[:DELTA_MASK_DIMS]
        state_stats.update(s.reshape(1, -1))
        action_stats.update(a_chunk)  # (H, 7) -> RunningStats reshapes to (-1, 7)

norm_stats = {
    'state': state_stats.get_statistics(),
    'actions': action_stats.get_statistics(),
}

output_path = pathlib.Path('./assets/${CONFIG_NAME}/${DATASET_REPO_ID}')
normalize.save(output_path, norm_stats)
print(f'  Norm stats saved to: {output_path}/norm_stats.json')
print(f'  State  mean: {norm_stats[\"state\"].mean}')
print(f'  Action mean: {norm_stats[\"actions\"].mean}')
print(f'  Action q01:  {norm_stats[\"actions\"].q01}')
print(f'  Action q99:  {norm_stats[\"actions\"].q99}')

# Sanity check: delta ranges should be tight after consistency fix
q01 = norm_stats['actions'].q01
q99 = norm_stats['actions'].q99
ax_range = float(q99[3] - q01[3])
ay_range = float(q99[4] - q01[4])
print(f'')
print(f'  [Sanity Check] ax delta range: {ax_range:.4f} rad')
print(f'  [Sanity Check] ay delta range: {ay_range:.4f} rad')
if ax_range > 1.0:
    print(f'  ⚠️  WARNING: ax delta range > 1.0 rad — axis-angle sign flips may still exist!')
else:
    print(f'  ✓  ax/ay delta ranges look normal (no sign flip inflation)')
"
fi

# ══════════════════════════════════════════════════════════════
#  Step 2: 启动 LoRA 微调训练
# ══════════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "  Step 2: Starting LoRA fine-tuning..."
echo "  Config:          ${CONFIG_NAME}"
echo "  Experiment:      ${EXP_NAME}"
echo "  Dataset:         ${DATASET_REPO_ID}"
echo "  GPU:             ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} device(s))"
echo "  Global batch:    ${BATCH_SIZE} (${BASE_BATCH_SIZE}/device × ${NUM_GPUS})"
echo "  Train steps:     ${NUM_TRAIN_STEPS}"
echo "  Peak LR:         ${PEAK_LR}"
echo "  Action horizon:  ${ACTION_HORIZON}"
echo "  Save interval:   ${SAVE_INTERVAL}"
echo "  Num workers:     ${NUM_WORKERS}"
echo "  FSDP devices:    ${FSDP_DEVICES}"
echo "  Checkpoint dir:  ${CHECKPOINT_DIR}/${CONFIG_NAME}/${EXP_NAME}/"
echo "============================================================"
echo ""

uv run python scripts/train.py "${CONFIG_NAME}" \
    --exp-name "${EXP_NAME}" \
    --num-train-steps ${NUM_TRAIN_STEPS} \
    --checkpoint-base-dir "${CHECKPOINT_DIR}" \
    --save-interval ${SAVE_INTERVAL} \
    $( [ -n "${MAX_CHECKPOINTS}" ] && echo "--max-checkpoints ${MAX_CHECKPOINTS}" ) \
    --batch-size ${BATCH_SIZE} \
    --model.action-horizon ${ACTION_HORIZON} \
    --fsdp-devices ${FSDP_DEVICES} \
    --num-workers ${NUM_WORKERS} \
    --ema-decay ${EMA_DECAY} \
    --seed ${SEED} \
    --log-interval ${LOG_INTERVAL} \
    --lr-schedule.warmup-steps ${WARMUP_STEPS} \
    --lr-schedule.peak-lr ${PEAK_LR} \
    --lr-schedule.decay-steps ${NUM_TRAIN_STEPS} \
    --lr-schedule.decay-lr ${DECAY_LR} \
    $( [ "${WANDB_ENABLED}" = true ] && echo "--wandb-enabled" || echo "--no-wandb-enabled" ) \
    $( if [ "${IMAGE_AUGMENT_ENABLED}" = true ]; then echo "--image-augment.enabled --image-augment.brightness ${IMAGE_AUGMENT_BRIGHTNESS} --image-augment.contrast ${IMAGE_AUGMENT_CONTRAST} --image-augment.saturation ${IMAGE_AUGMENT_SATURATION} --image-augment.hue ${IMAGE_AUGMENT_HUE}"; else echo "--no-image-augment.enabled"; fi )

echo ""
echo "============================================================"
echo "  Training complete!"
echo "  Checkpoints saved at: ${CHECKPOINT_DIR}/${CONFIG_NAME}/${EXP_NAME}/"
echo "============================================================"

