"""
本地推理配置文件
用于本地直接通信（无需SSH隧道）
"""

# ==================== 本地服务器配置 ====================
# 推理服务器配置（本地直接连接）
SERVER_IP = "127.0.0.1"      # 本地服务器地址
SERVER_PORT = 8007           # 观测上传端口
OBSERVATION_PORT = 8007
ACTION_PORT = 8008

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
IMAGE_QUALITY = 85           # JPEG 压缩质量 (1-100)
CAMERA_CONFIG_PATH = "control/cameras/camera_config.json"  # 相机配置文件路径
COLLECT_OBS_FREQ = 60.0

# ==================== 推理配置 ====================
# OpenPI Pi0.5 保持 10Hz 的动作 chunk 节拍，
# 但观测采集与控制回路保持更高频率以减少执行滞后。
INFERENCE_FREQ = 10.0
EXECUTION_FREQ = 60.0
EXECUTION_MODE = "rtc"
EXECUTE_HORIZON = 4
DELAY_ESTIMATE_ALPHA = 0.5
DELAY_ESTIMATE_INIT_STEPS = 2
MAX_DEADLINE_OVERRUN_STEPS = 2
ARRIVAL_SETTLE_POS_DELTA_THRESHOLD = 0.0005
ARRIVAL_SETTLE_STABLE_COUNT = 2
ARRIVAL_CHECK_FREQ = 60.0
ARRIVAL_LOG_INTERVAL_SEC = 0.5
ARRIVAL_MAX_WAIT_SEC = 1.5

# IK 可行性投影：目标位姿 IK 失败则沿路径回退搜索可行点
ENABLE_IK_PROJECTION = True
IK_LINE_SEARCH_STEPS = 10
IK_TOL = 1e-3

# ==================== 网络配置 ====================
SOCKET_TIMEOUT = 5.0         # Socket 超时时间 (秒)
BUFFER_SIZE = 4096           # 接收缓冲区大小 (字节)
ENCODING = 'utf-8'           # 字符编码
