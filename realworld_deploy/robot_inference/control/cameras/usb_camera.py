"""
USB 摄像头实现
基于 OpenCV，仅支持 RGB
"""
import cv2
import numpy as np
from typing import Dict, Any

from .base import BaseCamera


class USBCamera(BaseCamera):
    """USB 摄像头（OpenCV）"""
    
    def __init__(self, 
                 camera_index: int = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 **kwargs):
        """
        初始化 USB 摄像头
        
        Args:
            camera_index: 摄像头索引
            width: 图像宽度
            height: 图像高度
            fps: 帧率
        """
        super().__init__(width, height, fps)
        self.camera_index = camera_index
        self.cap = None
    
    def start(self) -> bool:
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"[USBCamera] ✗ 无法打开摄像头 {self.camera_index}")
                return False
            
            # 设置参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 获取实际参数
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            print(f"[USBCamera] ✓ 摄像头已启动")
            print(f"  索引: {self.camera_index}")
            print(f"  分辨率: {actual_width}x{actual_height}")
            print(f"  帧率: {actual_fps} fps")
            
            # 预热（清空缓冲区）
            self.clear_buffer()
            
            self._is_opened = True
            return True
            
        except Exception as e:
            print(f"[USBCamera] ✗ 启动失败: {e}")
            return False
    
    def clear_buffer(self, n_frames: int = 10):
        """清空缓冲区（重写基类方法，使用更快的 grab）"""
        if self.cap is None:
            return
        for _ in range(n_frames):
            self.cap.grab()
    
    def _read_frame_impl(self) -> Dict[str, Any]:
        """读取一帧"""
        if self.cap is None:
            return {'color': None, 'depth': None}
        
        ret, frame = self.cap.read()
        
        if ret:
            return {'color': frame, 'depth': None}
        else:
            return {'color': None, 'depth': None}
    
    def stop(self):
        """停止摄像头"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self._is_opened = False
        self.print_stats()
        print("[USBCamera] ✓ 已停止")
