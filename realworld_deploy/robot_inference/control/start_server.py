#!/usr/bin/env python3
"""
Polymetis 服务器启动脚本
连接到 Franka 硬件 (172.16.0.2)

使用方法：
    conda activate robot
    python start_server.py              # 只启动机器人服务器
    python start_server.py --gripper    # 同时启动机器人和夹爪服务器
    
可选参数：
    --sim       使用仿真模式（PyBullet）
    --readonly  只读模式（不发送力矩）
    --gripper   同时启动夹爪服务器（端口50052）
"""
import os
import sys
import subprocess
import time
import argparse
import getpass

def check_environment():
    """检查是否在正确的 conda 环境中"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != 'robot':
        print("⚠️  警告: 当前不在 robot conda 环境中")
        print(f"   当前环境: {conda_env if conda_env else '(base)'}")
        print("\n请先运行: conda activate robot")
        return False
    return True

def check_existing_server():
    """检查是否有服务器已在运行"""
    # 检查 launch_robot.py 进程
    result1 = subprocess.run(['pgrep', '-f', 'launch_robot.py'], 
                          capture_output=True)
    # 检查 run_server 进程（实际占用端口的进程）
    result2 = subprocess.run(['pgrep', '-f', 'run_server'], 
                          capture_output=True)
    # 检查端口占用
    result3 = subprocess.run(['lsof', '-i', ':50051'], 
                          capture_output=True)
    
    return (result1.returncode == 0 or 
            result2.returncode == 0 or 
            result3.returncode == 0)

def kill_existing_server():
    """终止已有的服务器进程"""
    print("检测到旧的服务器进程，正在清理...")
    print("提示：如需要输入密码，请输入管理员密码")
    
    # 先尝试按当前用户清理 run_server（避免使用 sudo）
    try:
        current_user = getpass.getuser()
    except Exception:
        current_user = None

    if current_user:
        result1 = subprocess.run(['pkill', '-9', '-u', current_user, 'run_server'],
                                 stderr=subprocess.DEVNULL)
    else:
        # 回退到不带用户限制的 pkill
        result1 = subprocess.run(['pkill', '-9', 'run_server'],
                                 stderr=subprocess.DEVNULL)
    
    # 清理 launch_robot.py（通常不需要 sudo）
    subprocess.run(['pkill', '-9', '-f', 'launch_robot.py'], 
                  stderr=subprocess.DEVNULL)
    
    # 再次检查清理是否成功
    time.sleep(2)
    
    # 验证进程已清理
    check1 = subprocess.run(['pgrep', '-f', 'run_server'], 
                           capture_output=True)
    check2 = subprocess.run(['pgrep', '-f', 'launch_robot.py'], 
                           capture_output=True)
    
    if check1.returncode == 0 or check2.returncode == 0:
        print("⚠️  警告：部分进程可能未完全清理")
        print("   如果启动失败，请手动运行: sudo pkill -9 run_server")
    else:
        print("✓ 清理完成\n")

def main():
    parser = argparse.ArgumentParser(description='启动Polymetis服务器')
    parser.add_argument('--sim', action='store_true', help='使用仿真模式')
    parser.add_argument('--readonly', action='store_true', help='只读模式')
    parser.add_argument('--robot-ip', type=str, default='172.16.0.2', 
                       help='机器人IP地址')
    # 以前通过 --gripper 控制是否启动夹爪服务器，现在硬件模式下默认总是启动
    args = parser.parse_args()
    
    print("=" * 70)
    print(" " * 20 + "Polymetis 服务器启动")
    print("=" * 70)
    print()
    
    # 检查环境
    if not check_environment():
        return 1
    
    print("✓ conda 环境正确 (robot)")
    print()
    
    # 实时模式可能需要额外权限（以前通过 sudo 获取），此处不再自动请求 sudo。
    print("注意: 实时模式可能需要额外权限以获得实时调度/优先级。")
    print()
    
    # 检查旧进程
    if check_existing_server():
        print("⚠️  检测到端口 50051 被占用或已有服务器在运行")
        response = input("是否自动清理并重启服务器？(y/n): ")
        if response.lower() == 'y':
            kill_existing_server()
        else:
            print("\n取消启动。如需手动清理，请运行:")
            print("  pkill -9 -u $(whoami) run_server  # 或使用 sudo 清理其他用户进程（如有必要）")
            return 0
    
    # 获取polymetis根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    polymetis_root = os.path.dirname(script_dir)  # /path/to/polymetis
    launch_script = os.path.join(
        polymetis_root, 
        'polymetis/python/scripts/launch_robot.py'
    )
    
    if not os.path.exists(launch_script):
        print(f"❌ 错误: 找不到启动脚本")
        print(f"   期望位置: {launch_script}")
        return 1
    
    # 构建启动命令
    if args.sim:
        # 仿真模式
        gripper_enabled = False
        mode = "仿真模式 (PyBullet)"
        cmd = [
            sys.executable,
            launch_script,
            'robot_client=bullet_sim',
            'use_real_time=false',
            'gui=false'
        ]
    else:
        # 硬件模式：总是启动夹爪服务器
        gripper_enabled = True
        mode = f"硬件模式 (IP: {args.robot_ip})"
        cmd = [
            sys.executable,
            launch_script,
            'robot_client=franka_hardware',
            f'robot_client.executable_cfg.robot_ip={args.robot_ip}'
        ]
        
        if args.readonly:
            cmd.append('robot_client.executable_cfg.readonly=true')
            mode += " [只读]"
    
    # 配置信息
    print("配置信息:")
    print(f"  模式:          {mode}")
    print(f"  机器人服务器:  0.0.0.0:50051")
    if gripper_enabled:
        print(f"  夹爪服务器:    0.0.0.0:50052")
    print(f"  启动脚本:      {launch_script}")
    print()
    
    print(f"工作目录: {polymetis_root}")
    print()
    print("-" * 70)
    if gripper_enabled:
        print("正在启动机器人服务器和夹爪服务器...")
    else:
        print("正在启动机器人服务器...")
    print("按 Ctrl+C 停止所有服务器")
    print("-" * 70)
    print()
    
    # 设置环境变量
    env = os.environ.copy()
    libfranka_path = '/home/kyji/Desktop/Workspace/libfranka/build'
    if 'LD_LIBRARY_PATH' in env:
        env['LD_LIBRARY_PATH'] = f"{libfranka_path}:{env['LD_LIBRARY_PATH']}"
    else:
        env['LD_LIBRARY_PATH'] = libfranka_path
    
    # 启动机器人服务器（后台运行）
    processes = []
    try:
        print("[1] 启动机器人服务器...")
        robot_process = subprocess.Popen(cmd, cwd=polymetis_root, env=env)
        processes.append(('robot', robot_process))
        print(f"✓ 机器人服务器已启动 (PID: {robot_process.pid})")
        time.sleep(3)  # 等待机器人服务器启动
        
        # 启动夹爪服务器（仅硬件模式）
        if gripper_enabled:
            print("\n[2] 启动夹爪服务器...")
            gripper_script = os.path.join(
                polymetis_root,
                'polymetis/python/scripts/launch_gripper.py'
            )
            # 注意：不要传递错误的 gripper.robot_ip 覆盖参数，
            # 直接使用配置文件中的默认 IP（通常已经是 172.16.0.2）
            gripper_cmd = [
                sys.executable,
                gripper_script,
                'gripper=franka_hand',
            ]
            gripper_process = subprocess.Popen(gripper_cmd, cwd=polymetis_root, env=env)
            processes.append(('gripper', gripper_process))
            print(f"✓ 夹爪服务器已启动 (PID: {gripper_process.pid})")
        
        print()
        print("=" * 70)
        print("✓ 所有服务器启动完成")
        print("=" * 70)
        print()
        
        # 等待所有进程
        for name, proc in processes:
            proc.wait()
            
    except KeyboardInterrupt:
        print()
        print()
        print("=" * 70)
        print("用户中断，正在关闭所有服务器...")
        print("=" * 70)
        
        # 清理所有进程（优先清理当前用户的 ）
        try:
            current_user = getpass.getuser()
        except Exception:
            current_user = None

        if current_user:
            subprocess.run(['pkill', '-9', '-u', current_user, 'run_server'],
                           stderr=subprocess.DEVNULL)
        else:
            subprocess.run(['pkill', '-9', 'run_server'],
                           stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-9', '-f', 'launch_robot.py'],
                       stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-9', '-f', 'launch_gripper.py'],
                       stderr=subprocess.DEVNULL)

        for name, proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except:
                proc.kill()

        print("✓ 所有服务器已关闭")
        return 0
    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ 服务器启动失败: {e}")
        print("=" * 70)
        print()
        if not args.sim:
            print("常见问题排查:")
            print("1. 检查机械臂是否开机且处于用户模式")
            print(f"2. 检查网络连接: ping {args.robot_ip}")
            print("3. 检查 libfranka 版本是否正确")
            print("4. 确保机械臂已从 Franka Desk 解锁")
            print("5. 确保外部激活设备（EAD）已释放（蓝灯）")
        
        # 清理进程
        for name, proc in processes:
            try:
                proc.terminate()
            except:
                pass
        return 1
    
    return 0

if __name__ == "__main__":
    # sudo pkill -9 run_server
    # sudo pkill -9 -f franka_panda_client
    # sudo pkill -9 -f franka_hand_client
    # pkill -9 -f launch_robot
    # pkill -9 -f launch_gripper
    sys.exit(main())
