"""
相机抽象基类
定义统一的相机接口
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
import time


class BaseCamera(ABC):
    """相机抽象基类"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._is_opened = False
        
        # 统计信息
        self.frames_captured = 0
        self.failed_reads = 0
        self.start_time = 0
    
    @abstractmethod
    def start(self) -> bool:
        """
        启动相机
        
        Returns:
            bool: 是否成功启动
        """
        pass
    
    @abstractmethod
    def _read_frame_impl(self) -> Dict[str, Any]:
        """
        读取一帧的具体实现（子类实现）
        
        Returns:
            dict: {'color': np.ndarray | None, 'depth': np.ndarray | None}
        """
        pass
    
    def read_frame(self) -> Dict[str, Any]:
        """
        读取一帧，返回统一格式
        
        Returns:
            dict: {
                'color': np.ndarray | None,  # BGR 图像 (H, W, 3)
                'depth': np.ndarray | None,  # 深度图 (H, W)，单位米
                'timestamp': float,          # 时间戳
            }
        """
        if not self._is_opened:
            return {'color': None, 'depth': None, 'timestamp': time.time()}
        
        try:
            result = self._read_frame_impl()
            result['timestamp'] = time.time()
            
            if result.get('color') is not None:
                self.frames_captured += 1
            else:
                self.failed_reads += 1
            
            return result
        except Exception as e:
            self.failed_reads += 1
            print(f"[Camera] 读取帧失败: {e}")
            return {'color': None, 'depth': None, 'timestamp': time.time()}
    
    @abstractmethod
    def stop(self):
        """停止相机"""
        pass
    
    @property
    def is_opened(self) -> bool:
        return self._is_opened
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self.frames_captured + self.failed_reads
        success_rate = (self.frames_captured / total * 100) if total > 0 else 0
        return {
            'frames_captured': self.frames_captured,
            'failed_reads': self.failed_reads,
            'success_rate': success_rate,
        }
    
    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print(f"\n[Camera] 统计信息:")
        print(f"  总帧数: {stats['frames_captured']}")
        print(f"  失败次数: {stats['failed_reads']}")
        print(f"  成功率: {stats['success_rate']:.1f}%")
    
    def clear_buffer(self, n_frames: int = 10):
        """
        清空缓冲区（子类可重写）
        
        Args:
            n_frames: 要丢弃的帧数
        """
        # 默认实现：读取并丢弃 n_frames 帧
        for _ in range(n_frames):
            self._read_frame_impl()
    
    def read_latest_frame(self) -> Dict[str, Any]:
        """
        读取最新帧（先清空缓冲区）
        
        用于需要严格时间同步的场景，确保返回的是当前时刻的帧。
        注意：此方法比 read_frame() 慢，因为需要先清空缓冲区。
        
        Returns:
            dict: 同 read_frame()
        """
        self.clear_buffer(n_frames=5)
        return self.read_frame()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
