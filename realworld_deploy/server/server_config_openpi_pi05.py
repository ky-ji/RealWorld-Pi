"""
OpenPI Pi0.5 推理服务器配置文件

用于 Franka 真机部署，与 inference_client.py 配合使用。
使用 OpenPI 仓库的标准推理流程 (policy_config.create_trained_policy)。

注意：
  - 本版本使用 OpenPI 的训练/推理管线，动作空间为 7D 轴角格式:
    [x, y, z, ax, ay, az, gripper] (axis-angle rotation)
  - 客户端发送的是 8D 四元数格式: [x, y, z, qx, qy, qz, qw, gripper] (XYZW 标准顺序)
  - 服务器内部进行 四元数 XYZW ↔ 轴角 的转换
  - 仅支持增量 (delta) 模式（与 OpenPI 训练一致，模型预测 delta，output transform 自动加 state）
"""

from pathlib import Path
import os


REALWORLD_PI_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_ROOT = REALWORLD_PI_ROOT / "checkpoints"

# =============================================================================
# 服务器配置
# =============================================================================
SERVER_IP = "0.0.0.0"              # 监听所有网卡
SERVER_PORT = 8007                 # 观测上传端口
OBSERVATION_PORT = 8007
ACTION_PORT = 8008

# =============================================================================
# 模型配置 (OpenPI)
# =============================================================================
# OpenPI 训练配置名称（对应 src/openpi/training/config.py 中的 TrainConfig.name）
CONFIG_NAME = "pi05_stack_bowls_lora"
#CONFIG_NAME = "pi05_assembly_things_lora" 
#CONFIG_NAME = "pi05_place_phone_lora"

# OpenPI checkpoint 目录（包含 params/, train_state/, assets/ 等）
#CHECKPOINT_DIR = str(DEFAULT_CHECKPOINT_ROOT / "assembly_things_lora_0209" / "pi05_assembly_things_lora" / "assembly_things_lora_v1" / "14999")
CHECKPOINT_DIR = os.environ.get(
    "OPENPI_CHECKPOINT_DIR",
    str(DEFAULT_CHECKPOINT_ROOT / "stack_bowls_lora_0208" / "pi05_stack_bowls_lora" / "stack_bowls_lora_v2" / "29999"),
)
#CHECKPOINT_DIR = str(DEFAULT_CHECKPOINT_ROOT / "place_phone_lora_0211" / "pi05_place_phone_lora" / "place_phone_lora_v2" / "19999")
# 推理设备（用于 PyTorch 模型加载，JAX 模型自动选择）
DEVICE = "cuda"

# 任务指令（prompt）
# toast
# pineapple bun
#chocolate
#TASK_PROMPT = "move the chocolate from the conveyor belt to the center of the plate"
#TASK_PROMPT = "Insert the pen into the pen holder"
TASK_PROMPT = "Stack three bowls together"
#TASK_PROMPT = "Place the phone on a stand"

# 模型输入参数
NUM_IMAGES = 2                     # Pi0.5 使用 Front View + Wrist View
ACTION_DIM = 7                     # 动作维度：[x, y, z, ax, ay, az, gripper] (7D 轴角格式)

# =============================================================================
# 推理配置
# =============================================================================
INFERENCE_FREQ = 10.0              # 推理频率 (Hz)
EXECUTION_MODE = os.environ.get("OPENPI_EXECUTION_MODE", "naive_async")
EXECUTE_HORIZON = int(os.environ.get("OPENPI_EXECUTE_HORIZON", "4"))
DELAY_ESTIMATE_ALPHA = float(os.environ.get("OPENPI_DELAY_ESTIMATE_ALPHA", "0.5"))
DELAY_ESTIMATE_INIT_STEPS = int(os.environ.get("OPENPI_DELAY_ESTIMATE_INIT_STEPS", "2"))
MAX_DEADLINE_OVERRUN_STEPS = int(os.environ.get("OPENPI_MAX_DEADLINE_OVERRUN_STEPS", "2"))

# Chunk 和推理参数
# 注意：OpenPI 的 action_horizon 在训练 config 中设为 10，即模型输出 (10, 7) 的 action chunk
TARGET_CHUNK_SIZE = 10             # 每次发送完整 action horizon，执行窗口由 client 的 execute_horizon 控制

# 动作放大系数（补偿控制器跟踪衰减）
# 增量模型预测的 delta 通常较小，需要放大才能有效移动
# OpenPI 的 AbsoluteActions transform 已将 delta 加上 state，得到绝对目标位置
# 放大逻辑：delta = action - state → amplified_action = state + delta * amplify
ACTION_AMPLIFY_POS = 1.0           # 位置放大系数，建议 3.0~5.0（1.0=不放大）
ACTION_AMPLIFY_ROT = 1.0           # 旋转放大系数，建议 1.0~1.5（1.0=不放大）

# ---- 动作安全限制 ----
ENABLE_ACTION_LIMIT = False         # 是否启用动作限制（True=推荐，False=原始输出）

# 速度限制: 单步最大位移/旋转
# 在 10Hz 控制频率下：
#   位置: 0.04 m/step → 最大线速度 40 cm/s
#   旋转: 0.08 rad/step → ~46 °/s
MAX_POS_DELTA_PER_STEP = 0.05      # 单步最大位移 (米)，建议 0.02~0.05
MAX_ROT_DELTA_PER_STEP = 0.08      # 单步最大旋转轴角向量范数 (弧度)，建议 0.05~0.15

# 加速度限制: 限制相邻帧之间速度变化
MAX_POS_ACCELERATION = 0.005       # 位置加速度限制 (米/step²)，None=不限制
MAX_ROT_ACCELERATION = 0.01        # 旋转加速度限制 (弧度/step²)，None=不限制

# 缓启动: Episode 前 N 步将速度限制线性从 10% 渐增到 100%
SMOOTH_START_STEPS = 5            # 缓启动步数，0=不缓启动

# EMA 平滑: 跨 chunk 边界的指数移动平均
ACTION_SMOOTHING_ALPHA = 0.2       # EMA 平滑系数，0.0=不平滑

# 夹爪 chunk 一致性保持
# 若 chunk 内夹爪二值化后不一致（部分开、部分闭），则整个 chunk 使用第一个 action 的夹爪值
# 防止 chunk 尾部的不确定切换导致下一步推理出错误的夹爪状态
GRIPPER_CHUNK_CONSISTENCY = False    # True=启用一致性保持，False=使用模型原始输出
GRIPPER_THRESHOLD = 0.5            # 夹爪二值化阈值：< threshold 视为闭合，>= threshold 视为打开

# 工作空间硬限制 (安全围栏，超出直接 clip)
SAFE_WORKSPACE = {
    'x': [0.2, 0.85],
    'y': [-0.5, 0.5],
    'z': [0.02, 0.65],
}

# =============================================================================
# 通信配置
# =============================================================================
SOCKET_TIMEOUT = 5.0               # Socket 超时 (秒)
BUFFER_SIZE = 40960                # 缓冲区大小
ENCODING = 'utf-8'                 # 编码格式
MAX_CLIENTS = 1                    # 最大客户端连接数

# =============================================================================
# 日志配置
# =============================================================================
VERBOSE = True                     # 是否打印详细日志
