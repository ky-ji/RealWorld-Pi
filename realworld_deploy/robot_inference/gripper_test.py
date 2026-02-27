#!/usr/bin/env python3
"""
夹爪独立测试脚本
输入 g 关闭夹爪，输入 k 打开夹爪，输入 q 退出
用于排查夹爪是否触发硬件保护
"""

import _path_setup

from inference_config_local import ROBOT_IP, GRIPPER_PORT

try:
    from polymetis import GripperInterface
except ImportError:
    print("✗ 无法导入 Polymetis，请确认环境")
    exit(1)


def main():
    print(f"[夹爪测试] 连接夹爪: {ROBOT_IP}:{GRIPPER_PORT}")
    gripper = GripperInterface(ip_address=ROBOT_IP, port=GRIPPER_PORT)
    print("[夹爪测试] ✓ 夹爪已连接")

    state = gripper.get_state()
    print(f"[夹爪测试] 当前宽度: {state.width:.4f}m")

    # 与 inference_client.py 完全一致的参数
    OPEN_PARAMS = dict(width=0.09, speed=0.3, force=1.0, blocking=True)
    GRASP_PARAMS = dict(speed=0.2, force=5.0, grasp_width=0.06,
                        epsilon_inner=0.05, epsilon_outer=0.05, blocking=True)

    print("\n操作说明:")
    print("  g  - 关闭夹爪 (grasp)")
    print("  k  - 打开夹爪 (goto)")
    print("  s  - 查询当前状态")
    print("  q  - 退出\n")

    while True:
        try:
            cmd = input(">>> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == 'g':
            print(f"[夹爪测试] 关闭夹爪... 参数: {GRASP_PARAMS}")
            try:
                gripper.grasp(**GRASP_PARAMS)
                state = gripper.get_state()
                print(f"[夹爪测试] ✓ 夹爪已关闭 (宽度: {state.width:.4f}m)")
            except Exception as e:
                print(f"[夹爪测试] ✗ 关闭失败: {e}")

        elif cmd == 'k':
            print(f"[夹爪测试] 打开夹爪... 参数: {OPEN_PARAMS}")
            try:
                gripper.goto(**OPEN_PARAMS)
                state = gripper.get_state()
                print(f"[夹爪测试] ✓ 夹爪已打开 (宽度: {state.width:.4f}m)")
            except Exception as e:
                print(f"[夹爪测试] ✗ 打开失败: {e}")

        elif cmd == 's':
            state = gripper.get_state()
            print(f"[夹爪测试] 当前宽度: {state.width:.4f}m")

        elif cmd == 'q':
            break

    print("[夹爪测试] 退出")


if __name__ == "__main__":
    main()
