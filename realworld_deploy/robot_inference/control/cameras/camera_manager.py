"""
多相机管理器
支持同时管理多个相机视角，统一读取和控制
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BaseCamera
from .usb_camera import USBCamera
from .realsense_camera import RealSenseCamera


class CameraManager:
    """多相机管理器"""

    def __init__(self, config_path: Optional[str] = None, cameras_config: Optional[List[Dict]] = None):
        """
        初始化多相机管理器

        Args:
            config_path: 相机配置文件路径（JSON格式）
            cameras_config: 相机配置列表（直接传入，优先级高于config_path）
                每个配置项格式：
                {
                    'name': 'front_view',  # 视角名称
                    'type': 'realsense',   # 相机类型
                    'serial_number': '123456',  # RealSense序列号（可选）
                    'camera_index': 0,     # USB相机索引（可选）
                    'width': 1280,
                    'height': 720,
                    'fps': 30,
                    'enable_depth': True   # 是否启用深度（RealSense）
                }
        """
        self.cameras: Dict[str, BaseCamera] = {}
        self.camera_names: List[str] = []

        # 加载配置
        if cameras_config is not None:
            config = cameras_config
        elif config_path is not None:
            config = self._load_config(config_path)
        else:
            raise ValueError("必须提供 config_path 或 cameras_config")

        # 创建相机实例
        for cam_config in config:
            name = cam_config['name']
            cam_type = cam_config['type'].lower()

            # 提取相机参数
            kwargs = {
                'width': cam_config.get('width', 640),
                'height': cam_config.get('height', 480),
                'fps': cam_config.get('fps', 30),
            }

            # 根据类型添加特定参数
            if cam_type in ('realsense', 'd435i', 'd435', 'rs'):
                kwargs['enable_depth'] = cam_config.get('enable_depth', True)
                kwargs['enable_color'] = cam_config.get('enable_color', True)
                kwargs['align_depth_to_color'] = cam_config.get('align_depth_to_color', True)
                if 'serial_number' in cam_config:
                    kwargs['serial_number'] = cam_config['serial_number']
                camera = RealSenseCamera(**kwargs)
            elif cam_type == 'usb':
                kwargs['camera_index'] = cam_config.get('camera_index', 0)
                camera = USBCamera(**kwargs)
            else:
                raise ValueError(f"不支持的相机类型: {cam_type}")

            self.cameras[name] = camera
            self.camera_names.append(name)

        print(f"[CameraManager] 已配置 {len(self.cameras)} 个相机视角:")
        for name in self.camera_names:
            print(f"  - {name}")

    def _load_config(self, config_path: str) -> List[Dict]:
        """从JSON文件加载相机配置"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 支持两种格式：直接列表 或 {'cameras': [...]}
        if isinstance(config, list):
            return config
        elif isinstance(config, dict) and 'cameras' in config:
            return config['cameras']
        else:
            raise ValueError("配置文件格式错误，应为列表或包含'cameras'键的字典")

    def start_all(self) -> Dict[str, bool]:
        """
        启动所有相机

        Returns:
            Dict[str, bool]: 每个相机的启动结果 {name: success}
        """
        print("\n[CameraManager] 启动所有相机...")
        results = {}

        for name, camera in self.cameras.items():
            print(f"\n启动相机: {name}")
            success = camera.start()
            results[name] = success

            if not success:
                print(f"⚠️  相机 {name} 启动失败")

        success_count = sum(results.values())
        print(f"\n[CameraManager] 启动完成: {success_count}/{len(self.cameras)} 个相机成功")

        return results

    def stop_all(self):
        """停止所有相机"""
        print("\n[CameraManager] 停止所有相机...")
        for name, camera in self.cameras.items():
            if camera.is_opened:
                camera.stop()
                print(f"✓ 相机 {name} 已停止")

    def read_all_frames(self, parallel: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        读取所有相机的帧

        Args:
            parallel: 是否并行读取（推荐，可提高效率）

        Returns:
            Dict[str, Dict]: {
                'front_view': {'color': np.ndarray, 'depth': np.ndarray, 'timestamp': float},
                'side_view': {...},
                ...
            }
        """
        if parallel:
            return self._read_frames_parallel()
        else:
            return self._read_frames_sequential()

    def _read_frames_sequential(self) -> Dict[str, Dict[str, Any]]:
        """顺序读取所有相机帧"""
        frames = {}
        for name, camera in self.cameras.items():
            if camera.is_opened:
                frames[name] = camera.read_frame()
            else:
                frames[name] = {'color': None, 'depth': None, 'timestamp': time.time()}
        return frames

    def _read_frames_parallel(self) -> Dict[str, Dict[str, Any]]:
        """并行读取所有相机帧（更快）"""
        frames = {}

        with ThreadPoolExecutor(max_workers=len(self.cameras)) as executor:
            # 提交所有读取任务
            future_to_name = {
                executor.submit(camera.read_frame): name
                for name, camera in self.cameras.items()
                if camera.is_opened
            }

            # 收集结果
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    frames[name] = future.result()
                except Exception as e:
                    print(f"⚠️  读取相机 {name} 失败: {e}")
                    frames[name] = {'color': None, 'depth': None, 'timestamp': time.time()}

        # 添加未启动的相机
        for name, camera in self.cameras.items():
            if not camera.is_opened and name not in frames:
                frames[name] = {'color': None, 'depth': None, 'timestamp': time.time()}

        return frames

    def read_latest_frames(self, parallel: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        读取所有相机的最新帧（先清空缓冲区）

        用于需要严格时间同步的场景

        Args:
            parallel: 是否并行读取

        Returns:
            Dict[str, Dict]: 同 read_all_frames()
        """
        # 先清空所有相机的缓冲区
        for camera in self.cameras.values():
            if camera.is_opened:
                camera.clear_buffer(n_frames=5)

        # 然后读取最新帧
        return self.read_all_frames(parallel=parallel)

    def clear_all_buffers(self, n_frames: int = 10):
        """清空所有相机的缓冲区"""
        for camera in self.cameras.values():
            if camera.is_opened:
                camera.clear_buffer(n_frames=n_frames)

    def get_camera(self, name: str) -> Optional[BaseCamera]:
        """获取指定名称的相机实例"""
        return self.cameras.get(name)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有相机的统计信息"""
        stats = {}
        for name, camera in self.cameras.items():
            stats[name] = camera.get_stats()
        return stats

    def print_all_stats(self):
        """打印所有相机的统计信息"""
        print("\n[CameraManager] 所有相机统计信息:")
        for name, camera in self.cameras.items():
            stats = camera.get_stats()
            print(f"\n  {name}:")
            print(f"    总帧数: {stats['frames_captured']}")
            print(f"    失败次数: {stats['failed_reads']}")
            print(f"    成功率: {stats['success_rate']:.1f}%")

    @property
    def is_any_opened(self) -> bool:
        """是否有任何相机已启动"""
        return any(camera.is_opened for camera in self.cameras.values())

    @property
    def all_opened(self) -> bool:
        """是否所有相机都已启动"""
        return all(camera.is_opened for camera in self.cameras.values())

    def __len__(self) -> int:
        """返回相机数量"""
        return len(self.cameras)

    def __enter__(self):
        """上下文管理器入口"""
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.stop_all()
