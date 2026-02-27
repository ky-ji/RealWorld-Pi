#!/usr/bin/env python3
"""
Franka 数据采集接收端
在 robot 环境中运行，接收 GELLO 数据并控制 Franka 机械臂，同时记录轨迹数据

使用方法：
    conda activate robot
    cd realworld_deploy/robot_inference/control
    python franka_data_collector.py
"""

import sys
import os
import time
import socket
import json
import numpy as np
import torch
import threading
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

# 设置路径（使代码可在任意目录运行）
# 获取当前文件所在目录
_CURRENT_DIR = Path(__file__).parent.resolve()
_ROBOT_INFERENCE_DIR = _CURRENT_DIR.parent
# 添加必要的路径
sys.path.insert(0, str(_ROBOT_INFERENCE_DIR))
sys.path.insert(0, str(_CURRENT_DIR))
sys.path.insert(0, str(_CURRENT_DIR / 'cameras'))

try:
    import polymetis
    from polymetis import RobotInterface
    from polymetis import GripperInterface
    print("✓ Polymetis 库导入成功")
except ImportError as e:
    print(f"✗ 无法导入 Polymetis 库: {e}")
    print("请确保在 robot conda 环境中运行")
    sys.exit(1)

from cameras import create_camera, CameraManager
from trajectory_episode import TrajectoryEpisode, get_next_episode_id

# 夹爪二元化阈值
GRIPPER_OPEN_THRESHOLD = 0.05  # 大于此值认为是打开状态（1），否则是关闭状态（0）


class FrankaDataCollector:
    """Franka 数据采集接收端"""
    
    def __init__(self, 
                 listen_host: str = "0.0.0.0",
                 listen_port: int = 5555,
                 control_port: int = 5556,
                 robot_ip: str = "localhost",
                 robot_port: int = 50051,
                 position_scale: float = 1.0,
                 save_dir: str = "data/trajectories",
                 camera_type: str = 'realsense',
                 camera_index: int = 0,
                 camera_width: int = 1280,
                 camera_height: int = 720,
                 enable_depth: bool = True,
                 camera_config_file: Optional[str] = None):
        """
        初始化 Franka 数据采集接收端
        
        Args:
            listen_host: GELLO数据监听地址
            listen_port: GELLO数据监听端口
            control_port: GUI控制端口
            robot_ip: Polymetis 服务器 IP
            robot_port: Polymetis 服务器端口
            position_scale: 位置映射缩放因子
            save_dir: 数据保存目录
            camera_type: 相机类型 ('realsense' 或 'usb')
            camera_index: 摄像头索引（仅 USB 相机）
            camera_width: 相机图像宽度
            camera_height: 相机图像高度
            enable_depth: 是否启用深度（仅 RealSense）
            camera_config_file: 多相机配置文件路径（JSON格式）
                               如果提供，将使用多相机模式，忽略单相机参数
        """
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.control_port = control_port
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.position_scale = position_scale
        
        # 数据保存
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查存储模式
        import os
        self.storage_mode = os.environ.get('STORAGE_MODE', 'simple')  # 默认为简单模式
        
        # Socket相关
        self.data_server_socket = None  # GELLO数据服务器
        self.data_client_socket = None  # GELLO数据客户端
        self.control_server_socket = None  # GUI控制服务器
        self.control_client_socket = None  # GUI控制客户端
        self.recv_buffer = b""
        
        # 机器人相关
        self.robot = None
        self.gripper = None
        self.initial_gello_joints = None
        self.initial_robot_joints = None
        
        # 控制线程相关
        self.current_target = None
        self.target_lock = threading.Lock()
        self.control_thread = None
        self.running = False
        
        # 夹爪控制
        self.gripper_toggle_state = 1  # 初始为打开状态
        self.last_gripper_button_state = 0
        self.gripper_width = 0.09  # 当前夹爪宽度 (初始为打开)
        self.last_gripper_command_time = 0  # 上次夹爪命令时间
        self.gripper_command_min_interval = 1.0  # 夹爪命令最小间隔（秒）- 增加到2秒避免崩溃
        self.gripper_lock = threading.Lock()  # 夹爪命令线程锁，避免并发冲突
        self.gripper_executing = False  # 标记夹爪是否正在执行命令
        
        # 关节偏置（增量控制模式，仅用于微调对齐）
        self.joint_offset = np.array([0.0, 0.10, -0.10, 0.0, 0.0, 0.0, 0.0])
        self.gello_joint_signs = np.array([1, 1, 1, 1, 1, 1, 1])
        
        # 相机（支持单相机和多相机模式）
        self.camera_config_file = camera_config_file
        self.multi_camera_mode = camera_config_file is not None

        if self.multi_camera_mode:
            # 多相机模式
            self.camera_manager = CameraManager(config_path=camera_config_file)
            self.camera = None  # 兼容性，保留但不使用
            self.camera_names = self.camera_manager.camera_names
        else:
            # 单相机模式（兼容旧代码）
            self.camera = create_camera(
                camera_type=camera_type,
                width=camera_width,
                height=camera_height,
                fps=30,  # RealSense 支持 6/15/30/60 Hz，不支持 10
                enable_depth=enable_depth,
                camera_index=camera_index,  # 仅 USB 相机使用
            )
            self.camera_manager = None
            self.camera_names = None
        
        # 数据采集
        self.is_recording = False
        self.recording_lock = threading.Lock()
        self.current_episode: Optional[TrajectoryEpisode] = None
        self.episode_count = get_next_episode_id(self.save_dir) - 1
        self.sample_counter = 0  # 采样计数器，用于降采样
        self.sample_rate_divider = 3  # 30Hz/3=10Hz 采集频率
        
        # 分层数据采集状态
        self.current_sample_id = None
        self.current_item_id = None
        self.current_phase = None  # 'initial_states', 'end_states', 'pick', 'place'
        self.current_episode_id = None  # 用于pick/place阶段的episode编号
        
        # 追踪最后保存的episode路径，用于删除功能
        self.last_saved_episode_path = None
        
        print("\n" + "="*60)
        print("Franka 数据采集接收端")
        print("="*60)
        print(f"GELLO数据监听: {listen_host}:{listen_port}")
        print(f"GUI控制端口: {control_port}")
        print(f"机器人服务器: {robot_ip}:{robot_port}")
        print(f"数据保存目录: {self.save_dir.absolute()}")
        print(f"下一个Episode: {self.episode_count + 1}")
        print(f"控制频率: 30 Hz | 采集频率: 10 Hz (1/{self.sample_rate_divider}降采样)")
        print("="*60 + "\n")
    
    def initialize(self) -> bool:
        """初始化机器人、摄像头和网络服务"""
        # 1. 连接机器人
        print("[1/4] 连接到 Polymetis 服务器...")
        try:
            self.robot = RobotInterface(
                ip_address=self.robot_ip,
                port=self.robot_port
            )
            print("✓ 已连接到机器人")
            
            # 连接夹爪
            try:
                self.gripper = GripperInterface(
                    ip_address=self.robot_ip,
                    port=50052
                )
                print("✓ 已连接到夹爪")
            except Exception as e:
                self.gripper = None
                print(f"ℹ️  夹爪服务器未启动，夹爪控制已禁用")
            
            # 读取初始关节位置
            self.initial_robot_joints = self.robot.get_joint_positions().numpy()
            print(f"  机器人当前关节: {np.round(self.initial_robot_joints, 3)}")
        except Exception as e:
            print(f"✗ 连接机器人失败: {e}")
            return False
        
        # 2. 启动摄像头
        print("\n[2/4] 启动摄像头...")
        if self.multi_camera_mode:
            # 多相机模式
            results = self.camera_manager.start_all()
            if not any(results.values()):
                print("⚠️  所有摄像头启动失败，将不记录图像")
        else:
            # 单相机模式
            if not self.camera.start():
                print("⚠️  摄像头启动失败，将不记录图像")

        # 3. 启动 GUI 控制服务器（先于 GELLO，避免 GUI 连接失败）
        print("\n[3/4] 启动 GUI 控制服务器...")
        try:
            self.control_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.control_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.control_server_socket.bind((self.listen_host, self.control_port))
            self.control_server_socket.listen(1)
            self.control_server_socket.settimeout(0.001)  # 1ms超时，避免阻塞主循环
            print(f"✓ GUI控制服务器已启动，监听 {self.listen_host}:{self.control_port}")
        except Exception as e:
            print(f"✗ 启动 GUI 控制服务器失败: {e}")
            return False

        # 4. 启动 GELLO 数据 Socket 服务器
        print("\n[4/4] 启动 GELLO 数据服务器...")
        try:
            self.data_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.data_server_socket.bind((self.listen_host, self.listen_port))
            self.data_server_socket.listen(1)
            print(f"✓ GELLO数据服务器已启动，监听 {self.listen_host}:{self.listen_port}")
            print("\n等待 GELLO 发送端连接...")
            
            # 等待客户端连接
            self.data_client_socket, client_address = self.data_server_socket.accept()
            print(f"✓ GELLO 发送端已连接: {client_address}")
            
            # 接收初始化数据
            init_data = self._receive_data()
            if init_data and init_data.get("type") == "init":
                gello_joints_all = np.array(init_data["initial_joints"])
                print(f"  GELLO 初始关节: {np.round(gello_joints_all, 3)}")
                self.initial_gello_joints = gello_joints_all[:7]
                
                if len(self.initial_gello_joints) != len(self.initial_robot_joints):
                    print(f"✗ 自由度不匹配")
                    return False
                
                print(f"✓ 自由度匹配: {len(self.initial_robot_joints)} DOF")
            else:
                print("✗ 未收到初始化数据")
                return False
                
        except Exception as e:
            print(f"✗ 启动 GELLO 数据服务器失败: {e}")
            return False

        print("\n" + "="*60)
        print("✓ 初始化完成")
        print("="*60 + "\n")

        # 5. 启动关节阻抗控制器
        print("启动关节阻抗控制器...")
        try:
            self.robot.start_joint_impedance()
            print("✓ 控制器已启动，等待就绪...")

            current_pos = self.robot.get_joint_positions()
            for i in range(10):
                time.sleep(0.5)
                try:
                    self.robot.update_desired_joint_positions(current_pos)
                    print("✓ 控制器就绪并激活")
                    break
                except Exception as e:
                    if i < 9:
                        print(f"  等待控制器就绪... ({i+1}/10)")
                    else:
                        raise Exception(f"控制器启动超时: {e}")
            
            with self.target_lock:
                self.current_target = current_pos.numpy()
            
            # 启动后台控制线程
            self.running = True
            self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()
            print("✓ 后台控制线程已启动")
            
            # 发送 ready 信号给 GELLO 端
            ready_signal = {"type": "ready"}
            self._send_data(ready_signal)
            print("✓ 已发送 ready 信号给 GELLO 端\n")
        except Exception as e:
            print(f"✗ 启动控制器失败: {e}")
            return False
        
        return True
    
    def _control_loop(self):
        """后台控制循环，持续发送控制命令（30 Hz）"""
        print("后台控制循环已启动，频率: 30 Hz")
        rate = 1.0 / 30.0
        
        while self.running:
            try:
                loop_start = time.time()
                
                # 获取当前目标位置并发送
                with self.target_lock:
                    if self.current_target is not None:
                        target = torch.from_numpy(self.current_target).float()
                        self.robot.update_desired_joint_positions(target)
                
                # 数据采集（如果正在录制）- 每3帧采集1次，实现10Hz
                if self.is_recording and self.current_episode is not None:
                    self.sample_counter += 1
                    if self.sample_counter >= self.sample_rate_divider:
                        self._collect_data_point()
                        self.sample_counter = 0
                
                # 控制循环频率
                elapsed = time.time() - loop_start
                sleep_time = rate - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                     
            except Exception as e:
                if not self.running:
                    break
                
                err_msg = str(e)
                print(f"控制循环错误: {err_msg}")
                
                # 如果控制器被服务器终止，尝试自动重启关节阻抗控制器
                if "no controller running" in err_msg or "start_joint_impedance" in err_msg:
                    try:
                        print("[CONTROL] 检测到控制器未运行，尝试重新启动关节阻抗控制器...")
                        self.robot.start_joint_impedance()
                        current_pos = self.robot.get_joint_positions()
                        with self.target_lock:
                            self.current_target = current_pos.numpy()
                        print("[CONTROL] 关节阻抗控制器已重新启动并激活")
                        # 重启后继续循环
                        continue
                    except Exception as e_restart:
                        print(f"[CONTROL] 重新启动关节阻抗控制器失败: {e_restart}")
                        break
                else:
                    # 其他未知错误，稍作延时避免疯狂重试
                    time.sleep(0.1)
    
    def _collect_data_point(self):
        """采集一个数据点"""
        try:
            timestamp = time.time() - self.current_episode.start_time

            # 采集图像和深度
            if self.multi_camera_mode:
                # 多相机模式：并行读取所有相机
                frames_dict = self.camera_manager.read_latest_frames(parallel=True)

                image_indices = {}
                depth_indices = {}

                for cam_name, frame_data in frames_dict.items():
                    if frame_data['color'] is not None:
                        image_indices[cam_name] = self.current_episode.save_image(
                            frame_data['color'], camera_name=cam_name
                        )
                    else:
                        image_indices[cam_name] = -1

                    if frame_data['depth'] is not None:
                        depth_indices[cam_name] = self.current_episode.save_depth(
                            frame_data['depth'], camera_name=cam_name
                        )
                    else:
                        depth_indices[cam_name] = -1

                # 多相机模式的数据点
                image_index = -1  # 兼容性
                depth_index = -1  # 兼容性
            else:
                # 单相机模式（兼容旧代码）
                image_index = -1
                depth_index = -1
                image_indices = None
                depth_indices = None

                if self.camera.is_opened:
                    frame_data = self.camera.read_latest_frame()
                    if frame_data['color'] is not None:
                        image_index = self.current_episode.save_image(frame_data['color'])
                    if frame_data['depth'] is not None:
                        depth_index = self.current_episode.save_depth(frame_data['depth'])

            # 读取当前末端位姿（观测）
            ee_pos, ee_quat = self.robot.get_ee_pose()
            # ee_pos: torch.Tensor (3,) [x, y, z]
            # ee_quat: torch.Tensor (4,) [qx, qy, qz, qw]
            robot_eef_pose = np.concatenate([
                ee_pos.cpu().numpy(),
                ee_quat.cpu().numpy()
            ])  # (7,)

            # 读取夹爪状态并二元化
            gripper_binary = 1 if self.gripper_width > GRIPPER_OPEN_THRESHOLD else 0

            # 计算目标末端位姿（动作）
            with self.target_lock:
                if self.current_target is not None:
                    # 使用正运动学计算目标末端位姿
                    target_joints = torch.from_numpy(self.current_target).float()
                    target_ee_pos, target_ee_quat = self.robot.robot_model.forward_kinematics(target_joints)
                    action = np.concatenate([
                        target_ee_pos.cpu().numpy(),
                        target_ee_quat.cpu().numpy()
                    ])  # (7,)
                else:
                    action = robot_eef_pose.copy()

            action_gripper = gripper_binary

            # 添加数据点
            self.current_episode.add_data_point(
                timestamp=timestamp,
                robot_eef_pose=robot_eef_pose,
                robot_gripper=gripper_binary,
                action=action,
                action_gripper=action_gripper,
                image_index=image_index,
                depth_index=depth_index,
                image_indices=image_indices,
                depth_indices=depth_indices
            )
        except Exception as e:
            print(f"⚠️  采集数据点失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _send_data(self, data: dict):
        """发送 JSON 数据到 GELLO 端"""
        json_str = json.dumps(data)
        message = (json_str + "\n").encode('utf-8')
        self.data_client_socket.sendall(message)
    
    def _receive_data(self) -> dict:
        """接收 JSON 数据从 GELLO 端"""
        try:
            while True:
                if b"\n" in self.recv_buffer:
                    line, self.recv_buffer = self.recv_buffer.split(b"\n", 1)
                    return json.loads(line.decode('utf-8'))
                
                chunk = self.data_client_socket.recv(4096)
                if not chunk:
                    return None
                self.recv_buffer += chunk
        except Exception as e:
            print(f"✗ 接收数据失败: {e}")
            return None
    
    def _handle_gui_commands(self):
        """处理GUI控制命令（非阻塞）"""
        if self.control_client_socket is None:
            # 尝试接受连接（非阻塞）
            try:
                self.control_client_socket, _ = self.control_server_socket.accept()
                self.control_client_socket.settimeout(0.001)  # 1ms超时，避免阻塞控制循环
                print("✓ GUI 控制端已连接")
            except socket.timeout:
                pass
            except Exception as e:
                pass
            return

        # 接收命令
        try:
            data = self.control_client_socket.recv(1024)
            if not data:
                print("GUI 控制端断开连接")
                self.control_client_socket = None
                return

            command = json.loads(data.decode('utf-8'))
            cmd_type = command.get('type')

            if cmd_type == 'start_sample':
                self.current_sample_id = command.get('sample_id')
                print(f"\n📦 开始Sample: {self.current_sample_id}")
            elif cmd_type == 'start_item':
                self.current_item_id = command.get('item_id')
                print(f"🔧 开始Item: {self.current_item_id}")
            elif cmd_type == 'start_recording':
                sample_id = command.get('sample_id', self.current_sample_id)
                item_id = command.get('item_id', self.current_item_id)
                phase = command.get('phase')
                episode_id = command.get('episode_id')
                self.start_recording(sample_id, item_id, phase, episode_id)
            elif cmd_type == 'stop_recording':
                self.stop_recording()
            elif cmd_type == 'get_status':
                self._send_status()
            elif cmd_type == 'delete_last_episode':
                self._delete_last_episode()

        except socket.timeout:
            pass
        except Exception as e:
            pass
    
    def _send_status(self):
        """发送状态给GUI"""
        if self.control_client_socket is None:
            return

        try:
            # 计算图像数量（兼容单相机和多相机模式）
            if self.current_episode:
                if self.current_episode.multi_camera_mode:
                    n_images = sum(self.current_episode.image_counts.values())
                else:
                    n_images = self.current_episode.image_count
            else:
                n_images = 0

            status = {
                'is_recording': self.is_recording,
                'episode_id': self.current_episode.episode_id if self.current_episode else None,
                'n_steps': len(self.current_episode.data_points) if self.current_episode else 0,
                'n_images': n_images,
                'duration': time.time() - self.current_episode.start_time if self.current_episode else 0.0,
            }
            data = json.dumps(status).encode('utf-8')
            self.control_client_socket.sendall(data)
        except Exception as e:
            pass
    
    def start_recording(self, sample_id=None, item_id=None, phase=None, episode_id=None):
        """开始记录新的episode"""
        with self.recording_lock:
            if self.is_recording:
                print(f"⚠️  已经在记录中")
                return
            
            # 清空摄像头缓冲区，确保录制的是最新画面
            if self.multi_camera_mode:
                self.camera_manager.clear_all_buffers(n_frames=10)
            elif self.camera.is_opened:
                self.camera.clear_buffer(n_frames=10)
            
            # 设置当前状态
            if sample_id:
                self.current_sample_id = sample_id
            if item_id:
                self.current_item_id = item_id
            self.current_phase = phase
            self.current_episode_id = episode_id
            
            # 构建保存路径
            if self.storage_mode == 'hierarchical' and sample_id:
                # 分层存储模式（用于 block_building GUI）
                save_path = self.save_dir / f"sample_{sample_id}"
                
                if phase == 'initial_states' or phase == 'end_states':
                    # 初始/结束状态：sample_X/initial_states/ 或 sample_X/end_states/
                    save_path = save_path / phase
                elif phase == 'pick' or phase == 'place':
                    # pick/place阶段：sample_X/pick/episode_1/ 或 sample_X/place/episode_1/
                    save_path = save_path / phase / episode_id
                else:
                    # 未知阶段，使用默认路径
                    save_path = self.save_dir
                
                # 在分层模式下，路径已经完整，不需要再创建episode_xxxx子文件夹
                use_subdirectory = False
                
            else:
                # 简单存储模式（用于普通 GUI）
                save_path = self.save_dir
                use_subdirectory = True  # 创建episode_xxxx子文件夹
            self.episode_count += 1
            self.current_episode = TrajectoryEpisode(
                self.episode_count,
                save_path,
                use_subdirectory=use_subdirectory,
                camera_names=self.camera_names  # 传入相机名称列表
            )
            self.sample_counter = 0  # 重置采样计数器
            self.is_recording = True
            
            # 显示录制信息
            if self.storage_mode == 'hierarchical':
                phase_str = f" - {phase}" if phase else ""
                episode_str = f" - {episode_id}" if episode_id else ""
                print(f"\n🔴 开始记录{phase_str}{episode_str}")
            else:
                print(f"\n🔴 开始记录 Episode {self.episode_count}")
            
            print(f"   保存路径: {self.current_episode.episode_folder}")
            print(f"   使用子文件夹: {use_subdirectory}")
    
    def stop_recording(self):
        """停止记录并保存当前episode"""
        with self.recording_lock:
            if not self.is_recording or self.current_episode is None:
                print(f"⚠️  没有正在记录的episode")
                return

            self.is_recording = False
            self.sample_counter = 0  # 重置采样计数器

            if len(self.current_episode.data_points) > 0:
                episode_folder, pkl_file, json_file = self.current_episode.save()
                n_steps = len(self.current_episode.data_points)
                # 兼容单相机和多相机模式
                if self.current_episode.multi_camera_mode:
                    n_images = sum(self.current_episode.image_counts.values())
                else:
                    n_images = self.current_episode.image_count
                duration = self.current_episode.data_points[-1]['timestamp']

                # 保存最后一个episode的路径，用于删除功能
                self.last_saved_episode_path = episode_folder

                phase_str = f" - {self.current_phase}" if self.current_phase else ""
                episode_str = f" - {self.current_episode_id}" if self.current_episode_id else ""

                print(f"✓ 录制完成{phase_str}{episode_str}")
                print(f"  数据点数: {n_steps}")
                print(f"  图像数: {n_images}")
                print(f"  持续时间: {duration:.2f}秒")
                print(f"  数据频率: {n_steps/duration:.1f} Hz")
                print(f"  保存位置: {episode_folder}")
            else:
                # 没有数据点，回退episode计数
                print(f"⚠️  录制的episode没有数据点，不保存")
                if self.episode_count > 0:
                    self.episode_count -= 1

            # 清除当前阶段信息（但保留sample_id和item_id）
            self.current_phase = None
            self.current_episode_id = None
    
    def _find_last_episode_folder(self) -> Optional[Path]:
        """
        从文件夹中查找最后一个有效的episode

        支持两种存储模式：
        - simple: save_dir/episode_xxxx/
        - hierarchical: save_dir/sample_X/pick/episode_X/ 或 sample_X/place/episode_X/

        Returns:
            最后一个episode的路径，如果没有则返回None
        """
        if not self.save_dir.exists():
            return None

        all_episodes = []

        if self.storage_mode == 'hierarchical':
            # 分层模式：递归搜索所有包含data.pkl的文件夹
            for sample_dir in self.save_dir.iterdir():
                if not sample_dir.is_dir() or not sample_dir.name.startswith('sample_'):
                    continue
                for phase_dir in sample_dir.iterdir():
                    if not phase_dir.is_dir() or phase_dir.name not in ['pick', 'place', 'initial_states', 'end_states']:
                        continue
                    for ep_dir in phase_dir.iterdir():
                        if ep_dir.is_dir() and (ep_dir / 'data.pkl').exists():
                            # 使用修改时间排序
                            mtime = (ep_dir / 'data.pkl').stat().st_mtime
                            all_episodes.append((mtime, ep_dir))
        else:
            # 简单模式：直接在save_dir下查找episode_xxxx
            for item in self.save_dir.iterdir():
                if item.is_dir() and item.name.startswith('episode_'):
                    if (item / 'data.pkl').exists() or (item / 'data.npz').exists():
                        data_file = item / 'data.pkl' if (item / 'data.pkl').exists() else item / 'data.npz'
                        mtime = data_file.stat().st_mtime
                        all_episodes.append((mtime, item))

        if not all_episodes:
            return None

        # 返回修改时间最新的episode
        all_episodes.sort(key=lambda x: x[0], reverse=True)
        return all_episodes[0][1]

    def _delete_last_episode(self):
        """删除最后一条保存的episode"""
        import shutil

        if self.control_client_socket is None:
            return

        response = {'success': False, 'message': ''}

        try:
            if self.is_recording:
                response['message'] = '正在录制中，无法删除'
                print("⚠️  正在录制中，无法删除")
            else:
                # 从文件夹中查找最后一个episode（更可靠）
                episode_path = self._find_last_episode_folder()

                if episode_path is None:
                    response['message'] = '没有可删除的episode'
                    print("⚠️  没有可删除的episode")
                else:
                    # 删除episode文件夹
                    shutil.rmtree(episode_path)
                    print(f"🗑️  已删除episode: {episode_path}")

                    response['success'] = True
                    response['message'] = f'已删除: {episode_path.name}'

                    # 清空记录（如果删除的是last_saved_episode_path）
                    if self.last_saved_episode_path and Path(self.last_saved_episode_path) == episode_path:
                        self.last_saved_episode_path = None

                    # 更新episode计数为当前最大编号
                    next_episode = self._find_last_episode_folder()
                    if next_episode:
                        self.episode_count = int(next_episode.name.split('_')[1])
                    else:
                        self.episode_count = 0

        except Exception as e:
            response['message'] = f'删除失败: {str(e)}'
            print(f"✗ 删除失败: {e}")
            import traceback
            traceback.print_exc()

        # 发送响应
        try:
            data = json.dumps(response).encode('utf-8')
            self.control_client_socket.sendall(data)
            print(f"📤 发送删除响应: {response}")
        except Exception as e:
            print(f"✗ 发送响应失败: {e}")
    
    def _execute_gripper_command(self, toggle_state, width, cmd_time):
        """在单独线程中执行夹爪命令，避免阻塞主循环"""
        with self.gripper_lock:
            if self.gripper_executing:
                return
            self.gripper_executing = True
        
        try:
            if toggle_state == 1:
                # 打开：使用 goto
                self.gripper.goto(
                    width=width,
                    speed=0.2,
                    force=1.0,
                    blocking=True
                )
            else:
                # 关闭/抓取：使用 grasp
                # 注意：force 过大会导致硬件保护，1.0N 足够柔软物体
                self.gripper.grasp(
                    speed=0.2,
                    force=1.0,
                    grasp_width=width,
                    epsilon_inner=0.1,
                    epsilon_outer=0.1,
                    blocking=True
                )
            
            print(f"[Gripper] ✓ 完成: {'打开' if toggle_state else '闭合'}")
            
        except Exception as e:
            print(f"[Gripper] ✗ 失败: {e}")
            # 失败时设置额外惩罚时间
            self.last_gripper_command_time = time.time() + 3.0
        
        finally:
            with self.gripper_lock:
                self.gripper_executing = False
    
    def gello_to_robot_joints(self, gello_joints: np.ndarray) -> np.ndarray:
        """将 GELLO 关节角度映射到机器人关节角度"""
        gello_joints_corrected = gello_joints * self.gello_joint_signs
        initial_gello_corrected = self.initial_gello_joints * self.gello_joint_signs
        gello_delta = gello_joints_corrected - initial_gello_corrected
        scaled_delta = gello_delta * self.position_scale
        robot_target = self.initial_robot_joints + scaled_delta + self.joint_offset
        
        # 关节限位
        joint_limits_lower = np.array([-2.85, -1.75, -2.85, -3.05, -2.85, -0.01, -2.85])
        joint_limits_upper = np.array([2.85, 1.75, 2.85, -0.08, 2.85, 3.70, 2.85])
        robot_target = np.clip(robot_target, joint_limits_lower, joint_limits_upper)
        
        return robot_target
    
    def run(self):
        """运行接收和控制循环"""
        print("\n" + "="*60)
        print("开始接收并控制")
        print("="*60)
        print("\n请使用 GUI 界面控制数据采集")
        print("按 Ctrl+C 停止...")
        print("="*60 + "\n")
        
        start_time = time.time()
        receive_count = 0
        last_print_time = start_time
        
        try:
            while True:
                # 1. 处理 GUI 命令
                self._handle_gui_commands()
                
                # 2. 接收 GELLO 数据
                data = self._receive_data()
                
                if not data:
                    print("\n✗ GELLO连接断开")
                    break
                
                if data["type"] == "stop":
                    print("\n收到停止信号")
                    break
                
                elif data["type"] == "joint_state":
                    gello_joints = np.array(data["joints"])
                    
                    # 分离关节和夹爪数据
                    if len(gello_joints) >= 8:
                        robot_joints = gello_joints[:7]
                        gripper_pos = float(gello_joints[7])
                    else:
                        robot_joints = gello_joints[:7]
                        gripper_pos = None
                    
                    # 转换为机器人目标
                    robot_target = self.gello_to_robot_joints(robot_joints)
                    
                    # 更新目标位置
                    with self.target_lock:
                        self.current_target = robot_target
                    
                    # 夹爪控制
                    if self.gripper is not None and gripper_pos is not None:
                        button_threshold = 0.5
                        button_state = 1 if gripper_pos >= button_threshold else 0
                        
                        # 仅在上升沿时切换状态并发送命令
                        if button_state == 1 and self.last_gripper_button_state == 0:
                            # 检查是否距离上次命令足够久（防抖）
                            current_time = time.time()
                            time_since_last_cmd = current_time - self.last_gripper_command_time
                            
                            if time_since_last_cmd >= self.gripper_command_min_interval:
                                self.gripper_toggle_state = 1 - self.gripper_toggle_state
                                
                                # 只在状态改变时才发送夹爪命令
                                max_open = 0.09
                                self.gripper_width = max_open * float(self.gripper_toggle_state)
                                
                                # 检查是否有其他命令正在执行
                                if self.gripper_executing:
                                    print(f"[Gripper] 忽略命令：夹爪正在执行中")
                                    # 更新时间戳，避免立即重试
                                    self.last_gripper_command_time = current_time
                                else:
                                    # 先更新时间戳，防止快速重复触发
                                    self.last_gripper_command_time = current_time
                                    
                                    # 异步执行夹爪命令（避免阻塞主循环）
                                    gripper_thread = threading.Thread(
                                        target=self._execute_gripper_command,
                                        args=(self.gripper_toggle_state, self.gripper_width, current_time),
                                        daemon=True
                                    )
                                    gripper_thread.start()
                                    print(f"[Gripper] 发送命令: {'打开' if self.gripper_toggle_state else '闭合'} ({self.gripper_width:.3f}m)")
                            else:
                                print(f"[Gripper] 忽略命令：距离上次命令仅{time_since_last_cmd:.1f}秒（需要>{self.gripper_command_min_interval}秒）")
                        
                        self.last_gripper_button_state = button_state
                    
                    receive_count += 1
                    
                    # 显示状态（每秒一次）
                    current_time = time.time()
                    if current_time - last_print_time >= 1.0:
                        elapsed = current_time - start_time
                        avg_rate = receive_count / elapsed if elapsed > 0 else 0
                        
                        recording_status = "🔴 录制中" if self.is_recording else "⚪ 待机"
                        print(f"[{elapsed:6.1f}s] {recording_status} | "
                              f"频率: {avg_rate:5.1f} Hz | "
                              f"目标: {np.round(robot_target[:3], 3)}")
                        
                        last_print_time = current_time
                
        except KeyboardInterrupt:
            print("\n\n检测到 Ctrl+C，正在停止...")
        finally:
            self.stop(start_time, receive_count)
    
    def stop(self, start_time: float, receive_count: int):
        """停止控制"""
        print("\n" + "="*60)
        print("停止控制")
        print("="*60)
        
        # 如果正在录制，先保存
        if self.is_recording:
            print("保存当前episode...")
            self.stop_recording()
        
        # 停止后台控制线程
        self.running = False
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
            print("✓ 后台控制线程已停止")
        
        # 停止机器人
        if self.robot:
            try:
                self.robot.terminate_current_policy()
                print("✓ 机器人已停止")
            except:
                pass
        
        # 停止摄像头
        if self.multi_camera_mode:
            self.camera_manager.stop_all()
        else:
            self.camera.stop()
        
        # 关闭连接
        if self.data_client_socket:
            self.data_client_socket.close()
        if self.data_server_socket:
            self.data_server_socket.close()
        if self.control_client_socket:
            self.control_client_socket.close()
        if self.control_server_socket:
            self.control_server_socket.close()
        
        print("✓ 所有连接已关闭")
        
        # 统计信息
        if start_time:
            total_time = time.time() - start_time
            avg_rate = receive_count / total_time if total_time > 0 else 0
            print(f"\n统计信息:")
            print(f"  总运行时间: {total_time:.2f} 秒")
            print(f"  总接收次数: {receive_count}")
            print(f"  平均频率: {avg_rate:.1f} Hz")
        
        print("\n✓ 数据采集端已停止\n")


def main():
    """主函数"""
    parser = ArgumentParser(description='Franka 数据采集接收端')
    parser.add_argument('--listen-host', default='0.0.0.0',
                       help='Socket 监听地址')
    parser.add_argument('--listen-port', type=int, default=5555,
                       help='GELLO数据监听端口')
    parser.add_argument('--control-port', type=int, default=5556,
                       help='GUI控制端口')
    parser.add_argument('--robot-ip', default='localhost',
                       help='Polymetis 服务器 IP')
    parser.add_argument('--robot-port', type=int, default=50051,
                       help='Polymetis 服务器端口')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='位置缩放因子')
    parser.add_argument('--save-dir', default='data/trajectories',
                       help='数据保存目录')
    parser.add_argument('--camera-type', default='realsense',
                       choices=['realsense', 'usb'],
                       help='相机类型: realsense 或 usb')
    parser.add_argument('--camera-index', type=int, default=0,
                       help='摄像头索引（仅 USB 相机）')
    parser.add_argument('--camera-width', type=int, default=1280,
                       help='相机图像宽度')
    parser.add_argument('--camera-height', type=int, default=720,
                       help='相机图像高度')
    parser.add_argument('--enable-depth', action='store_true', default=True,
                       help='启用深度采集（仅 RealSense）')
    parser.add_argument('--no-depth', action='store_true',
                       help='禁用深度采集')
    parser.add_argument('--camera-config', type=str, default=None,
                       help='多相机配置文件路径（JSON格式），如果提供则使用多相机模式')
    
    args = parser.parse_args()
    
    # 处理深度参数
    enable_depth = not args.no_depth
    
    # 创建采集端
    collector = FrankaDataCollector(
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        control_port=args.control_port,
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        position_scale=args.scale,
        save_dir=args.save_dir,
        camera_type=args.camera_type,
        camera_index=args.camera_index,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        enable_depth=enable_depth,
        camera_config_file=args.camera_config
    )
    
    # 初始化
    if not collector.initialize():
        print("\n✗ 初始化失败，退出\n")
        return
    
    # 运行
    collector.run()


if __name__ == '__main__':
    main()
