"""
推理配置文件
包含 SSH 隧道、机器人、摄像头等配置参数
"""

# ==================== SSH 隧道配置 ====================
SSH_HOST = "115.190.134.186"
SSH_USER = "jikangye"
SSH_PORT = 22

# 推理服务器配置
SERVER_PORT = 8007
LOCAL_PORT = 8007
OBSERVATION_PORT = 8007
ACTION_PORT = 8008
LOCAL_OBSERVATION_PORT = 8007
LOCAL_ACTION_PORT = 8008

# ==================== 机器人配置 ====================
ROBOT_IP = "localhost"
ROBOT_PORT = 50051
GRIPPER_PORT = 50052

CARTESIAN_KX = [400, 400, 400, 10, 10, 10]
CARTESIAN_KXD = [40, 40, 40, 3, 3, 3]

# ==================== 摄像头配置 ====================
IMAGE_QUALITY = 85
CAMERA_CONFIG_PATH = "control/cameras/camera_config.json"
COLLECT_OBS_FREQ = 60.0

# ==================== 推理配置 ====================
INFERENCE_FREQ = 10.0
EXECUTION_FREQ = 60.0
ARRIVAL_SETTLE_POS_DELTA_THRESHOLD = 0.0005
ARRIVAL_SETTLE_STABLE_COUNT = 2
ARRIVAL_CHECK_FREQ = 60.0
ARRIVAL_LOG_INTERVAL_SEC = 0.5
ARRIVAL_MAX_WAIT_SEC = 1.5

ENABLE_IK_PROJECTION = True
IK_LINE_SEARCH_STEPS = 10
IK_TOL = 1e-3

# ==================== 网络配置 ====================
SOCKET_TIMEOUT = 5.0
BUFFER_SIZE = 4096
ENCODING = 'utf-8'
