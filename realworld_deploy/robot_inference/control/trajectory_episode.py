"""
轨迹数据 Episode 管理模块
用于保存和管理数据采集的轨迹数据

支持两种存储模式：
- 简单模式：save_dir/episode_xxxx/
- 分层模式：save_dir/sample_X/phase/episode_X/
"""
import time
import pickle
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple


def get_next_episode_id(save_dir: Path) -> int:
    """
    获取下一个 episode ID
    
    扫描 save_dir 下所有的 episode_xxxx 文件夹，返回下一个可用的 ID
    
    Args:
        save_dir: 保存目录
    
    Returns:
        int: 下一个可用的 episode ID
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return 1
    
    max_id = 0
    for item in save_dir.iterdir():
        if item.is_dir() and item.name.startswith('episode_'):
            try:
                episode_id = int(item.name.split('_')[1])
                max_id = max(max_id, episode_id)
            except (IndexError, ValueError):
                continue
    
    return max_id + 1


class TrajectoryEpisode:
    """
    单个轨迹 episode 数据管理器
    
    支持：
    - 单相机/多相机图像保存
    - 深度图保存
    - 灵活的存储目录结构
    - 自动图像编号
    """
    
    def __init__(self, 
                 episode_id: int, 
                 save_dir: Path,
                 use_subdirectory: bool = True,
                 camera_names: Optional[List[str]] = None):
        """
        初始化 Episode
        
        Args:
            episode_id: Episode ID
            save_dir: 保存目录
            use_subdirectory: 是否创建 episode_xxxx 子目录
            camera_names: 多相机模式下的相机名称列表，None 表示单相机模式
        """
        self.episode_id = episode_id
        self.start_time = time.time()
        self.data_points = []
        self.save_dir = Path(save_dir)
        
        # 多相机支持
        self.camera_names = camera_names
        self.multi_camera_mode = camera_names is not None and len(camera_names) > 0
        
        # 创建 episode 文件夹
        if use_subdirectory:
            self.episode_folder = self.save_dir / f'episode_{self.episode_id:04d}'
        else:
            self.episode_folder = self.save_dir
        
        # 创建图像和深度目录
        if self.multi_camera_mode:
            # 多相机模式：为每个相机创建子目录
            self.images_folders = {}
            self.depth_folders = {}
            self.image_counts = {}
            self.depth_counts = {}
            
            for cam_name in camera_names:
                self.images_folders[cam_name] = self.episode_folder / 'images' / cam_name
                self.images_folders[cam_name].mkdir(parents=True, exist_ok=True)
                self.image_counts[cam_name] = 0
                
                self.depth_folders[cam_name] = self.episode_folder / 'depth' / cam_name
                self.depth_folders[cam_name].mkdir(parents=True, exist_ok=True)
                self.depth_counts[cam_name] = 0
        else:
            # 单相机模式
            self.images_folder = self.episode_folder / 'images'
            self.images_folder.mkdir(parents=True, exist_ok=True)
            self.image_count = 0
            
            self.depth_folder = self.episode_folder / 'depth'
            self.depth_folder.mkdir(parents=True, exist_ok=True)
            self.depth_count = 0
    
    def save_image(self, image: np.ndarray, camera_name: Optional[str] = None) -> int:
        """
        保存图像
        
        Args:
            image: 图像数据 (BGR格式)
            camera_name: 相机名称（多相机模式必须提供）
        
        Returns:
            int: 图像索引
        """
        if self.multi_camera_mode:
            if camera_name is None or camera_name not in self.images_folders:
                raise ValueError(f"多相机模式下必须提供有效的相机名称，可用: {list(self.images_folders.keys())}")
            
            folder = self.images_folders[camera_name]
            index = self.image_counts[camera_name]
            image_path = folder / f'frame_{index:04d}.jpg'
            cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self.image_counts[camera_name] = index + 1
            return index
        else:
            image_path = self.images_folder / f'frame_{self.image_count:04d}.jpg'
            cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            current_index = self.image_count
            self.image_count += 1
            return current_index
    
    def save_depth(self, depth: np.ndarray, camera_name: Optional[str] = None) -> int:
        """
        保存深度图
        
        Args:
            depth: 深度数据（通常是 uint16 或 float32）
            camera_name: 相机名称（多相机模式必须提供）
        
        Returns:
            int: 深度图索引
        """
        if self.multi_camera_mode:
            if camera_name is None or camera_name not in self.depth_folders:
                raise ValueError(f"多相机模式下必须提供有效的相机名称，可用: {list(self.depth_folders.keys())}")
            
            folder = self.depth_folders[camera_name]
            index = self.depth_counts[camera_name]
            depth_path = folder / f'depth_{index:04d}.npy'
            np.save(str(depth_path), depth)
            self.depth_counts[camera_name] = index + 1
            return index
        else:
            depth_path = self.depth_folder / f'depth_{self.depth_count:04d}.npy'
            np.save(str(depth_path), depth)
            current_index = self.depth_count
            self.depth_count += 1
            return current_index
    
    def add_data_point(self,
                       timestamp: float,
                       robot_eef_pose: np.ndarray,
                       robot_gripper: float,
                       action: np.ndarray,
                       action_gripper: float,
                       image_index: int = -1,
                       depth_index: int = -1,
                       image_indices: Optional[Dict[str, int]] = None,
                       depth_indices: Optional[Dict[str, int]] = None,
                       **extra_data):
        """
        添加数据点
        
        Args:
            timestamp: 相对时间戳（秒）
            robot_eef_pose: 机械臂末端位姿 [x, y, z, qx, qy, qz, qw] (7,)
            robot_gripper: 夹爪状态 (0=关闭, 1=打开)
            action: 动作 [x, y, z, qx, qy, qz, qw] (7,)
            action_gripper: 动作夹爪
            image_index: 单相机图像索引
            depth_index: 单相机深度索引
            image_indices: 多相机图像索引 {camera_name: index}
            depth_indices: 多相机深度索引 {camera_name: index}
            **extra_data: 额外数据
        """
        data_point = {
            'timestamp': timestamp,
            'robot_eef_pose': robot_eef_pose,
            'robot_gripper': robot_gripper,
            'action': action,
            'action_gripper': action_gripper,
            'image_index': image_index,
            'depth_index': depth_index,
        }
        
        # 多相机索引
        if image_indices is not None:
            data_point['image_indices'] = image_indices
        if depth_indices is not None:
            data_point['depth_indices'] = depth_indices
        
        # 额外数据
        data_point.update(extra_data)
        
        self.data_points.append(data_point)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        n_steps = len(self.data_points)
        
        if n_steps == 0:
            return {
                'episode_id': self.episode_id,
                'start_time': self.start_time,
                'duration': 0.0,
                'n_steps': 0,
            }
        
        # 基础数据
        episode_data = {
            'episode_id': self.episode_id,
            'start_time': self.start_time,
            'duration': self.data_points[-1]['timestamp'] if n_steps > 0 else 0.0,
            'n_steps': n_steps,
            
            # 时间戳
            'timestamp': np.array([d['timestamp'] for d in self.data_points]),
            
            # 机械臂状态
            'robot_eef_pose': np.array([d['robot_eef_pose'] for d in self.data_points]),
            'robot_gripper': np.array([d['robot_gripper'] for d in self.data_points]),
            
            # 动作
            'action': np.array([d['action'] for d in self.data_points]),
            'action_gripper': np.array([d['action_gripper'] for d in self.data_points]),
            
            # 图像索引
            'image_index': np.array([d['image_index'] for d in self.data_points]),
            'depth_index': np.array([d.get('depth_index', -1) for d in self.data_points]),
        }
        
        # 多相机模式
        if self.multi_camera_mode:
            episode_data['n_images'] = sum(self.image_counts.values())
            episode_data['n_depths'] = sum(self.depth_counts.values())
            episode_data['camera_names'] = self.camera_names
            episode_data['image_counts'] = self.image_counts
            episode_data['depth_counts'] = self.depth_counts
            
            # 多相机索引
            if 'image_indices' in self.data_points[0]:
                for cam_name in self.camera_names:
                    episode_data[f'image_index_{cam_name}'] = np.array([
                        d.get('image_indices', {}).get(cam_name, -1) for d in self.data_points
                    ])
            if 'depth_indices' in self.data_points[0]:
                for cam_name in self.camera_names:
                    episode_data[f'depth_index_{cam_name}'] = np.array([
                        d.get('depth_indices', {}).get(cam_name, -1) for d in self.data_points
                    ])
        else:
            episode_data['n_images'] = self.image_count
            episode_data['n_depths'] = self.depth_count
        
        return episode_data
    
    def save(self) -> Tuple[Path, Path, Path]:
        """
        保存 episode 数据
        
        Returns:
            Tuple: (episode_folder, pkl_file, json_file)
        """
        episode_data = self.to_dict()
        
        # 保存为 pickle 格式
        pkl_file = self.episode_folder / 'data.pkl'
        with open(pkl_file, 'wb') as f:
            pickle.dump(episode_data, f)
        
        # 保存元数据为 JSON
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist() if obj.size < 100 else f"shape={obj.shape}"
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(x) for x in obj]
            return obj
        
        meta_data = {
            'episode_id': self.episode_id,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'duration': episode_data.get('duration', 0.0),
            'n_steps': episode_data['n_steps'],
            'multi_camera_mode': self.multi_camera_mode,
        }
        
        if self.multi_camera_mode:
            meta_data['camera_names'] = self.camera_names
            meta_data['image_counts'] = self.image_counts
            meta_data['depth_counts'] = self.depth_counts
        else:
            meta_data['n_images'] = getattr(self, 'image_count', 0)
            meta_data['n_depths'] = getattr(self, 'depth_count', 0)
        
        json_file = self.episode_folder / 'meta.json'
        with open(json_file, 'w') as f:
            json.dump(meta_data, f, indent=2, default=convert_for_json)
        
        return self.episode_folder, pkl_file, json_file


__all__ = ['TrajectoryEpisode', 'get_next_episode_id']




