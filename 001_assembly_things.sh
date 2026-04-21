#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  Assembly Things - Pi0.5 LoRA Fine-tuning Script (Multi-Task)
#  Tasks: 3 tasks (e.g., stack bowls, place phone, insert pen)
#  Model: Pi0.5 with LoRA (JAX, multi-GPU data parallel)
#  Action space: 7D [x, y, z, ax, ay, az, gripper] (axis-angle)
#
#  Usage:
#    bash 001_assembly_things.sh              # 默认使用 GPU 0
#    bash 001_assembly_things.sh 3            # 使用 GPU 3
#    bash 001_assembly_things.sh 0,1,2,3     # 使用 GPU 0,1,2,3 (四卡并行)
#    bash 001_assembly_things.sh 4,5,6,7     # 使用 GPU 4,5,6,7
# ══════════════════════════════════════════════════════════════

set -e  # Exit on error
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# ──────────────────────── GPU 配置 ────────────────────────
# 从命令行参数读取 GPU ID，默认使用 GPU 0
GPU_IDS="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# 自动计算 GPU 数量（按逗号分隔计数）
NUM_GPUS=$(echo "${GPU_IDS}" | awk -F',' '{print NF}')

# ──────────────────────── 环境配置 ────────────────────────
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
cd "${REPO_ROOT}"

# ──────────────────────── 训练参数配置 ────────────────────────

# 配置名称（对应 config.py 中注册的训练配置）
CONFIG_NAME="pi05_assembly_things_lora"

# 实验名称（用于区分不同训练运行，checkpoint 目录的子文件夹名）
EXP_NAME="assembly_things_lora_v1"

# 数据集 repo_id（对应 /data1/vla-data/processed/PI/data/ 下的目录名）
DATASET_REPO_ID="001_assembly_things_0209"

# 数据集路径
DATA_DIR="/data1/vla-data/processed/PI/data/${DATASET_REPO_ID}"

# ---- 单卡基准参数（以单卡为基准定义，多卡时自动缩放） ----
# 数据集约 290,000 个有效样本 (stack_bowls 的 6 倍), batch_size=32,
# 每 epoch 约 9,062 步。50,000 步 ≈ 5.5 epochs。
# 多任务数据量大、多样性高，不需要太多 epoch 就能收敛。
# 建议范围: 40,000 ~ 60,000 (4~7 epochs)
BASE_NUM_GPUS=1
BASE_BATCH_SIZE=32           # 单卡 batch size
BASE_NUM_TRAIN_STEPS=60000   # 单卡总训练步数
BASE_PEAK_LR="3e-5"          # 单卡峰值学习率

# ---- 多卡自动缩放 ----
# 全局 batch_size = 单卡 batch × GPU 数量（per-device batch 保持不变）
# 训练步数 = 单卡步数 / GPU 数量（保持总 epoch 数不变）
# 学习率 = 单卡 LR × sqrt(GPU 数量)（线性缩放太激进，sqrt 更稳定）
BATCH_SIZE=$((BASE_BATCH_SIZE * NUM_GPUS))
NUM_TRAIN_STEPS=$((BASE_NUM_TRAIN_STEPS / NUM_GPUS))

# 学习率 sqrt 缩放（用 python 计算浮点数）
PEAK_LR=$(python3 -c "import math; print(f'{${BASE_PEAK_LR} * math.sqrt(${NUM_GPUS}):.1e}')")
DECAY_LR=$(python3 -c "import math; print(f'{3e-6 * math.sqrt(${NUM_GPUS}):.1e}')")
WARMUP_STEPS=$((2000 / NUM_GPUS > 100 ? 2000 / NUM_GPUS : 100))

# ---- Checkpoint 配置 ----
# Checkpoint 输出根目录。最终路径: ${CHECKPOINT_DIR}/${CONFIG_NAME}/${EXP_NAME}/<step>/
CHECKPOINT_DIR="./checkpoints/assembly_things_lora_0209"

# Checkpoint 保存间隔（按步数缩放以保持相同 epoch 间隔）
SAVE_INTERVAL=$((10000 / NUM_GPUS > 500 ? 10000 / NUM_GPUS : 500))

# Checkpoint 保留策略: step % KEEP_PERIOD == 0 的 checkpoint 永久保留
KEEP_PERIOD=${SAVE_INTERVAL}

# ---- Action Chunk Size (动作预测步数) ----
# ⚠️ 注意: compute_norm_stats.py 读取 config.py 中硬编码的 action_horizon，
#    如果修改此值，需同步修改 config.py 中 pi05_assembly_things_lora 配置的
#    action_horizon，并删除已缓存的 norm_stats.json 重新计算。
ACTION_HORIZON=10

# ---- 其他训练参数 ----
EMA_DECAY=0.999
SEED=42
LOG_INTERVAL=200        # 数据量大，适当增大日志间隔

# ---- FSDP 配置 (模型并行) ----
# LoRA 参数量很小，不需要 FSDP 模型切分，纯数据并行即可。
# 仅当显存不足时才考虑增大 FSDP_DEVICES（必须能整除 NUM_GPUS）。
FSDP_DEVICES=1

# ---- W&B 日志 ----
WANDB_ENABLED=true

# ──────────────────────── 打印配置摘要 ────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Multi-GPU Configuration Summary"
echo "══════════════════════════════════════════════════════════════"
echo "  GPU IDs:                ${GPU_IDS}"
echo "  Number of GPUs:         ${NUM_GPUS}"
echo "  Per-device batch size:  ${BASE_BATCH_SIZE}"
echo "  Global batch size:     ${BATCH_SIZE}"
echo "  Train steps:            ${NUM_TRAIN_STEPS} (base: ${BASE_NUM_TRAIN_STEPS} / ${NUM_GPUS} GPUs)"
echo "  Peak LR:                ${PEAK_LR} (base: ${BASE_PEAK_LR} × √${NUM_GPUS})"
echo "  Decay LR:               ${DECAY_LR}"
echo "  Warmup steps:           ${WARMUP_STEPS}"
echo "  Save interval:          ${SAVE_INTERVAL}"
echo "  FSDP devices:           ${FSDP_DEVICES}"
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
"
fi

# ══════════════════════════════════════════════════════════════
#  Step 2: 启动 LoRA 微调训练
# ══════════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "  Step 2: Starting Multi-Task LoRA fine-tuning..."
echo "  Config:          ${CONFIG_NAME}"
echo "  Experiment:      ${EXP_NAME}"
echo "  Dataset:         ${DATASET_REPO_ID}"
echo "  GPU:             ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} device(s))"
echo "  Global batch:    ${BATCH_SIZE} (${BASE_BATCH_SIZE}/device × ${NUM_GPUS})"
echo "  Train steps:     ${NUM_TRAIN_STEPS}"
echo "  Peak LR:         ${PEAK_LR}"
echo "  Action horizon:  ${ACTION_HORIZON}"
echo "  Save interval:   ${SAVE_INTERVAL}"
echo "  FSDP devices:    ${FSDP_DEVICES}"
echo "  Checkpoint dir:  ${CHECKPOINT_DIR}/${CONFIG_NAME}/${EXP_NAME}/"
echo "============================================================"
echo ""

uv run python scripts/train.py "${CONFIG_NAME}" \
    --exp-name "${EXP_NAME}" \
    --num-train-steps ${NUM_TRAIN_STEPS} \
    --checkpoint-base-dir "${CHECKPOINT_DIR}" \
    --save-interval ${SAVE_INTERVAL} \
    --keep-period ${KEEP_PERIOD} \
    --batch-size ${BATCH_SIZE} \
    --model.action-horizon ${ACTION_HORIZON} \
    --fsdp-devices ${FSDP_DEVICES} \
    --ema-decay ${EMA_DECAY} \
    --seed ${SEED} \
    --log-interval ${LOG_INTERVAL} \
    --lr-schedule.warmup-steps ${WARMUP_STEPS} \
    --lr-schedule.peak-lr ${PEAK_LR} \
    --lr-schedule.decay-steps ${NUM_TRAIN_STEPS} \
    --lr-schedule.decay-lr ${DECAY_LR} \
    $( [ "${WANDB_ENABLED}" = true ] && echo "--wandb-enabled" || echo "--no-wandb-enabled" ) \
    --overwrite

echo ""
echo "============================================================"
echo "  Training complete!"
echo "  Checkpoints saved at: ${CHECKPOINT_DIR}/${CONFIG_NAME}/${EXP_NAME}/"
echo "============================================================"
