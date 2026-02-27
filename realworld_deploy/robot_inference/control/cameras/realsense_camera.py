"""
Intel RealSense D435i 相机实现
支持 RGB + 深度
"""
import numpy as np
from typing import Dict, Any, Optional

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[RealSenseCamera] ⚠️ pyrealsense2 未安装")

from .base import BaseCamera


class RealSenseCamera(BaseCamera):
    """Intel RealSense D435i 相机"""
    
    def __init__(self,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 enable_depth: bool = True,
                 enable_color: bool = True,
                 align_depth_to_color: bool = True,
                 serial_number: Optional[str] = None,
                 **kwargs):
        """
        初始化 RealSense 相机
        
        Args:
            width: 图像宽度
            height: 图像高度
            fps: 帧率
            enable_depth: 是否启用深度流
            enable_color: 是否启用彩色流
            align_depth_to_color: 是否将深度图对齐到彩色图
            serial_number: 相机序列号（用于多相机场景）
        """
        super().__init__(width, height, fps)
        self.enable_depth = enable_depth
        self.enable_color = enable_color
        self.align_depth_to_color = align_depth_to_color
        self.serial_number = serial_number
        
        self.pipeline = None
        self.config = None
        self.align = None
        self.profile = None
        self.depth_scale = 1.0
    
    def start(self) -> bool:
        """启动相机"""
        if not REALSENSE_AVAILABLE:
            print("[RealSenseCamera] ✗ pyrealsense2 未安装")
            return False
        
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # 如果指定了序列号，启用特定设备
            if self.serial_number:
                self.config.enable_device(self.serial_number)
            
            # 配置深度流
            if self.enable_depth:
                self.config.enable_stream(
                    rs.stream.depth,
                    self.width,
                    self.height,
                    rs.format.z16,
                    self.fps
                )
            
            # 配置彩色流
            if self.enable_color:
                self.config.enable_stream(
                    rs.stream.color,
                    self.width,
                    self.height,
                    rs.format.bgr8,
                    self.fps
                )
            
            # 启动管道
            self.profile = self.pipeline.start(self.config)
            
            # 设置深度对齐
            if self.align_depth_to_color and self.enable_depth and self.enable_color:
                self.align = rs.align(rs.stream.color)
            
            # 获取深度比例
            if self.enable_depth:
                depth_sensor = self.profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
            
            # 获取设备信息
            device = self.profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            device_serial = device.get_info(rs.camera_info.serial_number)
            
            print(f"[RealSenseCamera] ✓ 相机已启动")
            print(f"  设备: {device_name}")
            print(f"  序列号: {device_serial}")
            print(f"  分辨率: {self.width}x{self.height}")
            print(f"  帧率: {self.fps} fps")
            print(f"  深度比例: {self.depth_scale}")
            
            # 预热相机
            print("[RealSenseCamera] 预热中...")
            for _ in range(30):
                self.pipeline.wait_for_frames()
            
            self._is_opened = True
            print("[RealSenseCamera] ✓ 预热完成")
            return True
            
        except Exception as e:
            print(f"[RealSenseCamera] ✗ 启动失败: {e}")
            return False
    
    def _read_frame_impl(self) -> Dict[str, Any]:
        """读取一帧"""
        if self.pipeline is None:
            return {'color': None, 'depth': None}
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # 对齐深度到彩色
            if self.align is not None:
                frames = self.align.process(frames)
            
            result = {'color': None, 'depth': None}
            
            # 获取彩色图像
            if self.enable_color:
                color_frame = frames.get_color_frame()
                if color_frame:
                    result['color'] = np.asanyarray(color_frame.get_data())
            
            # 获取深度图像（转换为米）
            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    result['depth'] = depth_image.astype(np.float32) * self.depth_scale
            
            return result
            
        except Exception as e:
            print(f"[RealSenseCamera] 读取帧失败: {e}")
            return {'color': None, 'depth': None}
    
    def clear_buffer(self, n_frames: int = 10):
        """
        清空缓冲区，确保下一次读取是最新帧
        
        使用 poll_for_frames() 非阻塞方式清空队列中所有旧帧，
        确保后续 wait_for_frames() 获取的是真正的最新帧。
        
        Args:
            n_frames: 最大清空帧数（防止无限循环）
        """
        if self.pipeline is None:
            return
        
        try:
            # 使用 try_wait_for_frames 非阻塞清空队列
            for _ in range(n_frames * 2):
                success, _ = self.pipeline.try_wait_for_frames(timeout_ms=1)
                if not success:
                    break  # 队列已空
        except Exception:
            pass
    
    def get_intrinsics(self) -> Dict[str, Any]:
        """
        获取相机内参
        
        Returns:
            dict: 相机内参
        """
        if not self._is_opened or self.profile is None:
            return {}
        
        try:
            if self.enable_color:
                stream = self.profile.get_stream(rs.stream.color)
            elif self.enable_depth:
                stream = self.profile.get_stream(rs.stream.depth)
            else:
                return {}
            
            intrinsics = stream.as_video_stream_profile().get_intrinsics()
            return {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'cx': intrinsics.ppx,
                'cy': intrinsics.ppy,
                'distortion_model': str(intrinsics.model),
                'distortion_coeffs': list(intrinsics.coeffs),
            }
        except Exception as e:
            print(f"[RealSenseCamera] 获取内参失败: {e}")
            return {}
    
    def stop(self):
        """停止相机"""
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        
        self._is_opened = False
        self.print_stats()
        print("[RealSenseCamera] ✓ 已停止")
