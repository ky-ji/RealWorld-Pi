"""
相机模块
提供统一的相机接口，支持多种相机类型
"""
from .base import BaseCamera
from .usb_camera import USBCamera
from .realsense_camera import RealSenseCamera
from .camera_manager import CameraManager


def create_camera(camera_type: str, **kwargs) -> BaseCamera:
    """
    工厂函数：创建相机实例
    
    Args:
        camera_type: 相机类型
            - 'usb': USB 摄像头（OpenCV）
            - 'realsense': Intel RealSense D435i
        **kwargs: 相机配置参数
            - width: 图像宽度（默认 640）
            - height: 图像高度（默认 480）
            - fps: 帧率（默认 30）
            - camera_index: USB 摄像头索引（仅 usb）
            - enable_depth: 是否启用深度（仅 realsense）
            - serial_number: 相机序列号（仅 realsense，用于多相机）
    
    Returns:
        BaseCamera: 相机实例
    
    Example:
        # USB 摄像头
        camera = create_camera('usb', camera_index=0, width=1920, height=1080)
        
        # RealSense 相机
        camera = create_camera('realsense', width=640, height=480, enable_depth=True)
        
        # 使用
        camera.start()
        frame_data = camera.read_frame()
        color = frame_data['color']  # BGR 图像
        depth = frame_data['depth']  # 深度图（米），USB 相机为 None
        camera.stop()
    """
    camera_type = camera_type.lower()
    
    if camera_type == 'usb':
        return USBCamera(**kwargs)
    elif camera_type in ('realsense', 'd435i', 'd435', 'rs'):
        return RealSenseCamera(**kwargs)
    else:
        raise ValueError(f"未知的相机类型: {camera_type}，支持: usb, realsense")


__all__ = [
    'BaseCamera',
    'USBCamera',
    'RealSenseCamera',
    'CameraManager',
    'create_camera',
]
