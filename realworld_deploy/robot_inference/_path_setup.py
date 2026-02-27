"""
路径设置模块
动态设置 Python 路径，使代码可以在任意目录下运行

使用方法：
    # 在需要导入的文件顶部添加：
    import _path_setup
"""
import os
import sys
from pathlib import Path


def setup_paths():
    """设置 Python 路径"""
    # 获取当前文件所在目录（robot_inference）
    ROBOT_INFERENCE_DIR = Path(__file__).parent.resolve()
    
    # 获取 realworld_deploy 目录
    REALWORLD_DEPLOY_DIR = ROBOT_INFERENCE_DIR.parent
    
    # 需要添加到 sys.path 的目录
    paths_to_add = [
        str(ROBOT_INFERENCE_DIR),           # robot_inference/
        str(ROBOT_INFERENCE_DIR / 'configs'),  # robot_inference/configs/
        str(ROBOT_INFERENCE_DIR / 'control'),  # robot_inference/control/
        str(ROBOT_INFERENCE_DIR / 'control' / 'cameras'),  # robot_inference/control/cameras/
    ]
    
    # 添加到 sys.path（如果不存在）
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return ROBOT_INFERENCE_DIR, REALWORLD_DEPLOY_DIR


def get_project_root():
    """获取项目根目录（robot_inference）"""
    return Path(__file__).parent.resolve()


def get_keys_dir():
    """获取 keys 目录"""
    return Path(__file__).parent.resolve() / 'keys'


def get_ssh_key_path(key_name: str = 'id_server') -> str:
    """获取 SSH 密钥的绝对路径"""
    key_path = get_keys_dir() / key_name
    if key_path.exists():
        return str(key_path)
    else:
        raise FileNotFoundError(f"SSH 密钥不存在: {key_path}")


def get_log_dir() -> Path:
    """获取日志目录路径，如果不存在则创建"""
    log_dir = ROBOT_INFERENCE_DIR / 'log'
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# 自动设置路径
ROBOT_INFERENCE_DIR, REALWORLD_DEPLOY_DIR = setup_paths()

# 导出常用路径
__all__ = [
    'setup_paths',
    'get_project_root', 
    'get_keys_dir',
    'get_ssh_key_path',
    'get_log_dir',
    'ROBOT_INFERENCE_DIR',
    'REALWORLD_DEPLOY_DIR'
]

