#!/usr/bin/env python3
"""
GR00T 格式适配器

负责:
- 将机器人观测转换为 GR00T VLA 输入格式
- 将 GR00T 动作输出转换回机器人控制格式
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple


class Gr00tFormatAdapter:
    """
    GR00T 格式适配器
    
    适配 Franka Panda 机器人的观测和动作格式
    
    GR00T 观测格式:
        observation = {
            "video": {
                "camera_name": np.ndarray,  # Shape: (B, T, H, W, 3), dtype: uint8
            },
            "state": {
                "state_name": np.ndarray,   # Shape: (B, T, D), dtype: float32
            },
            "language": {
                "task": [[str]],            # Shape: (B, 1)
            }
        }
    
    GR00T 动作格式:
        action = {
            "action_name": np.ndarray,  # Shape: (B, T, D), dtype: float32
        }
    """
    
    # 夹爪常量
    GRIPPER_OPEN = 1.0
    GRIPPER_CLOSED = 0.0
    GRIPPER_THRESHOLD = 0.5
    
    def __init__(
        self,
        camera_keys: List[str] = None,
        state_keys: Dict[str, int] = None,
        action_keys: Dict[str, int] = None,
        n_obs_steps: int = 2,
        language_key: str = "annotation.human.task_description",
    ):
        """
        初始化适配器
        
        Args:
            camera_keys: 相机名称列表，例如 ["wrist_cam", "front_cam"]
            state_keys: 状态键和维度的字典，例如 {"robot_eef_pose": 7, "robot_gripper": 1}
            action_keys: 动作键和维度的字典，例如 {"single_arm": 7, "gripper": 1}
            n_obs_steps: 观测历史步数
            language_key: 语言指令的键名
        """
        # 默认配置 (Franka Panda)
        self.camera_keys = camera_keys or ["wrist_cam"]
        self.state_keys = state_keys or {
            "robot_eef_pose": 7,  # [x, y, z, qx, qy, qz, qw]
            "robot_gripper": 1,   # [gripper_state]
        }
        self.action_keys = action_keys or {
            "single_arm": 7,      # [x, y, z, qx, qy, qz, qw]
            "gripper": 1,         # [gripper_action]
        }
        self.n_obs_steps = n_obs_steps
        self.language_key = language_key
    
    def obs_to_policy_input(
        self,
        images,  # 可以是 np.ndarray 或 Dict[str, np.ndarray]
        poses: np.ndarray,
        grippers: np.ndarray,
        task_instruction: str = "complete the task",
    ) -> Dict[str, Any]:
        """
        将原始观测转换为 GR00T 策略输入格式
        
        Args:
            images: 图像数据，支持两种格式：
                - 单相机: np.ndarray (n_obs_steps, H, W, 3), uint8
                - 多相机: Dict[str, np.ndarray] {camera_name: (n_obs_steps, H, W, 3)}
            poses: 位姿序列 (n_obs_steps, 7), float32
            grippers: 夹爪序列 (n_obs_steps, 1), float32
            task_instruction: 任务指令
        
        Returns:
            GR00T 格式的观测字典
        """
        # 添加 batch 维度 (B=1)
        # video: (B=1, T, H, W, 3)
        # state: (B=1, T, D)
        # language: [[str]] (B=1, 1)
        
        obs = {
            "video": {},
            "state": {},
            "language": {
                self.language_key: [[task_instruction]]
            }
        }
        
        # 处理图像 - 支持单相机和多相机两种格式
        if isinstance(images, dict):
            # 多相机格式: {camera_name: (n_obs_steps, H, W, 3)}
            for cam_key in self.camera_keys:
                if cam_key in images:
                    cam_images = images[cam_key]
                    if cam_images.ndim == 3:
                        # 单帧图像 (H, W, 3) -> (1, 1, H, W, 3)
                        cam_images = cam_images[np.newaxis, np.newaxis, ...]
                    elif cam_images.ndim == 4:
                        # 图像序列 (T, H, W, 3) -> (1, T, H, W, 3)
                        cam_images = cam_images[np.newaxis, ...]
                    obs["video"][cam_key] = cam_images.astype(np.uint8)
        else:
            # 单相机格式: (n_obs_steps, H, W, 3) - 兼容旧代码
            if images.ndim == 3:
                # 单帧图像 (H, W, 3) -> (1, 1, H, W, 3)
                images = images[np.newaxis, np.newaxis, ...]
            elif images.ndim == 4:
                # 图像序列 (T, H, W, 3) -> (1, T, H, W, 3)
                images = images[np.newaxis, ...]
            
            # 将相同图像分配给所有相机（兼容模式）
            for cam_key in self.camera_keys:
                obs["video"][cam_key] = images.astype(np.uint8)
        
        # 处理状态
        if poses.ndim == 1:
            # 单步位姿 (7,) -> (1, 1, 7)
            poses = poses[np.newaxis, np.newaxis, ...]
        elif poses.ndim == 2:
            # 位姿序列 (T, 7) -> (1, T, 7)
            poses = poses[np.newaxis, ...]
        
        if grippers.ndim == 1:
            # 单步夹爪 (1,) -> (1, 1, 1)
            grippers = grippers[np.newaxis, np.newaxis, ...]
        elif grippers.ndim == 2:
            # 夹爪序列 (T, 1) -> (1, T, 1)
            grippers = grippers[np.newaxis, ...]
        
        # 按配置分配状态键
        state_keys_list = list(self.state_keys.keys())
        if len(state_keys_list) >= 2:
            obs["state"][state_keys_list[0]] = poses.astype(np.float32)
            obs["state"][state_keys_list[1]] = grippers.astype(np.float32)
        else:
            # 合并为单个状态
            full_state = np.concatenate([poses, grippers], axis=-1)
            obs["state"][state_keys_list[0]] = full_state.astype(np.float32)
        
        return obs
    
    def policy_output_to_action(
        self,
        action_dict: Dict[str, np.ndarray],
        step_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        将 GR00T 策略输出转换为机器人动作
        
        Args:
            action_dict: GR00T 动作字典 {"single_arm": (B, T, 7), "gripper": (B, T, 1)}
            step_idx: 要执行的时间步索引
        
        Returns:
            pose_7d: 目标位姿 (7,) [x, y, z, qx, qy, qz, qw]
            gripper_value: 夹爪值 (1,)
            gripper_open: 夹爪是否打开
        """
        action_keys_list = list(self.action_keys.keys())
        
        # 提取位姿动作
        pose_key = action_keys_list[0]  # 通常是 "single_arm"
        if pose_key in action_dict:
            pose_action = action_dict[pose_key]
            # (B, T, D) -> (D,)
            pose_7d = pose_action[0, step_idx, :].astype(np.float32)
        else:
            pose_7d = np.zeros(7, dtype=np.float32)
        
        # 提取夹爪动作
        if len(action_keys_list) >= 2:
            gripper_key = action_keys_list[1]  # 通常是 "gripper"
            if gripper_key in action_dict:
                gripper_action = action_dict[gripper_key]
                gripper_value = gripper_action[0, step_idx, :].astype(np.float32)
            else:
                gripper_value = np.array([self.GRIPPER_OPEN], dtype=np.float32)
        else:
            gripper_value = np.array([self.GRIPPER_OPEN], dtype=np.float32)
        
        # 判断夹爪状态
        gripper_open = float(gripper_value[0]) > self.GRIPPER_THRESHOLD
        
        return pose_7d, gripper_value, gripper_open
    
    def get_action_sequence(
        self,
        action_dict: Dict[str, np.ndarray],
        max_steps: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray, bool]]:
        """
        获取完整的动作序列
        
        Args:
            action_dict: GR00T 动作字典
            max_steps: 最大步数限制
        
        Returns:
            动作列表 [(pose_7d, gripper_value, gripper_open), ...]
        """
        # 获取动作时间维度
        first_key = list(action_dict.keys())[0]
        action_horizon = action_dict[first_key].shape[1]
        
        if max_steps is not None:
            action_horizon = min(action_horizon, max_steps)
        
        actions = []
        for t in range(action_horizon):
            action = self.policy_output_to_action(action_dict, step_idx=t)
            actions.append(action)
        
        return actions
    
    @staticmethod
    def polymetis_state_to_obs(
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        gripper_open: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 Polymetis 状态转换为观测格式
        
        Args:
            ee_pos: 末端执行器位置 (3,)
            ee_quat: 末端执行器四元数 (4,) [qx, qy, qz, qw]
            gripper_open: 夹爪是否打开
        
        Returns:
            pose_7d: (7,) [x, y, z, qx, qy, qz, qw]
            gripper_1d: (1,) [gripper_state]
        """
        pose_7d = np.concatenate([ee_pos, ee_quat]).astype(np.float32)
        gripper_value = Gr00tFormatAdapter.GRIPPER_OPEN if gripper_open else Gr00tFormatAdapter.GRIPPER_CLOSED
        gripper_1d = np.array([gripper_value], dtype=np.float32)
        return pose_7d, gripper_1d
    
    @staticmethod
    def obs_to_polymetis_action(
        pose_7d: np.ndarray,
        gripper_1d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        将观测格式转换为 Polymetis 动作格式
        
        Args:
            pose_7d: (7,) [x, y, z, qx, qy, qz, qw]
            gripper_1d: (1,) [gripper_value]
        
        Returns:
            ee_pos: (3,)
            ee_quat: (4,)
            gripper_open: bool
        """
        pose_7d = np.array(pose_7d).flatten()
        ee_pos = pose_7d[:3].astype(np.float32)
        ee_quat = pose_7d[3:7].astype(np.float32)
        gripper_value = float(gripper_1d[0]) if isinstance(gripper_1d, np.ndarray) else float(gripper_1d)
        gripper_open = gripper_value > Gr00tFormatAdapter.GRIPPER_THRESHOLD
        return ee_pos, ee_quat, gripper_open


class Gr00tFrankaAdapter(Gr00tFormatAdapter):
    """
    Franka Panda 机器人专用适配器
    
    默认配置符合 Franka 机器人的状态和动作空间
    """
    
    def __init__(
        self,
        camera_keys: List[str] = None,
        n_obs_steps: int = 2,
    ):
        super().__init__(
            camera_keys=camera_keys or ["wrist_cam"],
            state_keys={
                "robot_eef_pose": 7,  # [x, y, z, qx, qy, qz, qw]
                "robot_gripper": 1,   # [gripper_state]
            },
            action_keys={
                "single_arm": 7,      # [x, y, z, qx, qy, qz, qw]
                "gripper": 1,         # [gripper_action]
            },
            n_obs_steps=n_obs_steps,
        )


class Gr00tSO100Adapter(Gr00tFormatAdapter):
    """
    SO100 / SO101 机器人专用适配器
    
    兼容 LeRobot 的 SO100 配置
    """
    
    ROBOT_STATE_KEYS = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    
    def __init__(
        self,
        camera_keys: List[str] = None,
        n_obs_steps: int = 1,
    ):
        super().__init__(
            camera_keys=camera_keys or ["front", "wrist"],
            state_keys={
                "single_arm": 5,      # 5个关节
                "gripper": 1,         # 夹爪
            },
            action_keys={
                "single_arm": 5,
                "gripper": 1,
            },
            n_obs_steps=n_obs_steps,
        )
    
    def obs_to_policy_input(
        self,
        robot_obs: Dict[str, Any],
        task_instruction: str = "complete the task",
    ) -> Dict[str, Any]:
        """
        将 SO100 机器人观测转换为 GR00T 格式
        
        Args:
            robot_obs: 机器人观测字典，包含相机帧和关节状态
            task_instruction: 任务指令
        
        Returns:
            GR00T 格式的观测字典
        """
        obs = {
            "video": {},
            "state": {},
            "language": {
                self.language_key: [[task_instruction]]
            }
        }
        
        # 处理相机
        for cam_key in self.camera_keys:
            if cam_key in robot_obs:
                img = robot_obs[cam_key]
                if img.ndim == 3:
                    img = img[np.newaxis, np.newaxis, ...]
                obs["video"][cam_key] = img.astype(np.uint8)
        
        # 处理关节状态
        state = np.array([robot_obs[k] for k in self.ROBOT_STATE_KEYS], dtype=np.float32)
        single_arm = state[:5][np.newaxis, np.newaxis, ...]  # (1, 1, 5)
        gripper = state[5:6][np.newaxis, np.newaxis, ...]     # (1, 1, 1)
        
        obs["state"]["single_arm"] = single_arm
        obs["state"]["gripper"] = gripper
        
        return obs
    
    def decode_action_chunk(
        self,
        action_dict: Dict[str, np.ndarray],
        step_idx: int = 0,
    ) -> Dict[str, float]:
        """
        将 GR00T 动作解码为 SO100 关节命令
        
        Returns:
            字典: {"shoulder_pan.pos": val, ...}
        """
        single_arm = action_dict["single_arm"][0, step_idx, :]  # (5,)
        gripper = action_dict["gripper"][0, step_idx, :]        # (1,)
        
        full = np.concatenate([single_arm, gripper], axis=0)    # (6,)
        
        return {
            joint_name: float(full[i])
            for i, joint_name in enumerate(self.ROBOT_STATE_KEYS)
        }

