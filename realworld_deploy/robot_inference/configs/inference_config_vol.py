"""
推理配置文件
包含 SSH 隧道、机器人、摄像头等配置参数
"""

# ==================== SSH 隧道配置 ====================
# SSH 服务器配置
SSH_HOST = "115.190.134.186"  # SSH 服务器地址
SSH_USER = "jikangye"            # SSH 用户名
# SSH_KEY 现在通过 _path_setup.get_ssh_key_path() 动态获取，无需在此配置
SSH_PORT = 22           # SSH 端口

# 推理服务器配置
SERVER_PORT = 8007         # 远程推理服务器端口（远程服务器上推理服务监听的端口）
LOCAL_PORT = 8007          # 本地转发端口

# ==================== 机器人配置 ====================
# Polymetis 服务器配置
ROBOT_IP = "localhost"       # Polymetis 服务器 IP
ROBOT_PORT = 50051           # Polymetis 机器人端口
GRIPPER_PORT = 50052         # Polymetis 夹爪端口

# 笛卡尔阻抗控制参数（控制机械臂响应速度）
# Kx: 笛卡尔刚度 [x, y, z, rx, ry, rz] - 数值越小，速度越慢
# Kxd: 笛卡尔阻尼 [x, y, z, rx, ry, rz] - 数值越大，运动越平滑
#
# 速度档位参考：
#   快速模式（默认）:  Kx=[750, 750, 750, 15, 15, 15],  Kxd=[37, 37, 37, 2, 2, 2]
#   中速模式（推荐）:  Kx=[400, 400, 400, 10, 10, 10],  Kxd=[40, 40, 40, 3, 3, 3]
#   慢速模式（安全）:  Kx=[200, 200, 200, 8, 8, 8],     Kxd=[45, 45, 45, 4, 4, 4]
#   超慢速（调试）:    Kx=[100, 100, 100, 5, 5, 5],     Kxd=[50, 50, 50, 5, 5, 5]
CARTESIAN_KX = [400, 400, 400, 10, 10, 10]   # 笛卡尔刚度（中速模式）
CARTESIAN_KXD = [40, 40, 40, 3, 3, 3]         # 笛卡尔阻尼

# ==================== 摄像头配置 ====================
CAMERA_TYPE = 'realsense'    # 相机类型: 'realsense' 或 'usb'
CAMERA_INDEX = 0             # 摄像头索引（仅 USB 相机使用）
CAMERA_SERIAL_NUMBER = '944622073019'  # RealSense 相机序列号（用于多相机场景，对应 front_view）
CAMERA_RESOLUTION = (1280, 720)  # 摄像头分辨率 (width, height)
IMAGE_QUALITY = 80           # JPEG 压缩质量 (1-100)
ENABLE_DEPTH = True          # 是否启用深度（仅 RealSense 相机）

# ==================== 推理配置 ====================
INFERENCE_FREQ = 10.0        # 推理频率 (Hz)
EXECUTION_MODE = "naive_async"
EXECUTE_HORIZON = 4
DELAY_ESTIMATE_ALPHA = 0.5
DELAY_ESTIMATE_INIT_STEPS = 2
MAX_DEADLINE_OVERRUN_STEPS = 2
N_OBS_STEPS = 2              # 观测步数（历史帧数）
CAMERA_FREQ = 30.0           # 摄像头采样频率 (Hz)
STEPS_PER_INFERENCE = 6    # 每次推理执行的动作数量（1 = 最实时，6 = 官方默认值）

# ==================== 动作执行配置 ====================
# 动作缩放因子 (0.0-1.0)
# - 1.0: 完全执行模型输出的动作（最快，需要模型准确）
# - 0.5: 执行50%的动作幅度（较安全）
# - 0.2: 执行20%的动作幅度（最安全，适合测试）
ACTION_SCALE = 1

# ==================== 动作后处理配置（Diffusion Policy / VLA 推荐） ====================
# Receding-horizon：每次预测一段动作序列，只执行前 K 步，其余丢弃等待下一次重规划
EXECUTE_ACTION_PREFIX = 4

# Temporal ensembling：对目标位姿做 EMA 平滑（0 表示关闭）
TEMPORAL_ENSEMBLE_ALPHA = 0.2

# IK 可行性投影：目标位姿 IK 失败则沿路径回退搜索可行点
ENABLE_IK_PROJECTION = True
IK_LINE_SEARCH_STEPS = 6
IK_TOL = 1e-3

# 回退点过滤（对 diffusion 的细小动作序列容易误伤，建议保持关闭）
ENABLE_BACKTRACK_FILTER = False

# 夹爪参数（注意：代码中已硬编码正确参数，这些配置暂不使用）
GRIPPER_OPEN_WIDTH = 0.09    # 夹爪打开宽度 (m) - 使用 goto()
GRIPPER_CLOSED_WIDTH = 0.0   # 夹爪关闭宽度 (m) - 使用 grasp()
GRIPPER_SPEED = 0.3          # 夹爪移动速度 (m/s)
GRIPPER_FORCE = 1.0          # 夹爪力 (N)

# ==================== 网络配置 ====================
SOCKET_TIMEOUT = 5.0         # Socket 超时时间 (秒)
BUFFER_SIZE = 4096           # 接收缓冲区大小 (字节)
ENCODING = 'utf-8'           # 字符编码

# ==================== 日志配置 ====================
SAVE_TRAJECTORY_LOG = True   # 是否保存轨迹日志
LOG_DIR = "log"              # 日志保存目录（相对于 robot_inference 目录）
