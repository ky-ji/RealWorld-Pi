#!/usr/bin/env python3
"""
查找连接的 RealSense 相机序列号
用于配置多相机系统
"""
import sys

try:
    import pyrealsense2 as rs
except ImportError:
    print("✗ 未安装 pyrealsense2")
    print("请运行: pip install pyrealsense2")
    sys.exit(1)


def list_realsense_cameras():
    """列出所有连接的 RealSense 相机"""
    print("=" * 60)
    print("查找 RealSense 相机")
    print("=" * 60)

    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("\n✗ 未找到任何 RealSense 设备")
        print("\n请检查:")
        print("  1. 相机是否已连接")
        print("  2. USB 线缆是否正常")
        print("  3. 是否有足够的 USB 带宽")
        return []

    print(f"\n✓ 找到 {len(devices)} 个 RealSense 设备:\n")

    camera_info = []
    for i, dev in enumerate(devices):
        try:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            firmware = dev.get_info(rs.camera_info.firmware_version)
            usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)

            print(f"设备 {i+1}:")
            print(f"  名称:     {name}")
            print(f"  序列号:   {serial}")
            print(f"  固件版本: {firmware}")
            print(f"  USB 类型: {usb_type}")
            print()

            camera_info.append({
                'index': i,
                'name': name,
                'serial_number': serial,
                'firmware': firmware,
                'usb_type': usb_type
            })
        except Exception as e:
            print(f"设备 {i+1}: 读取信息失败 - {e}\n")

    return camera_info


def generate_config(camera_info):
    """生成配置文件模板"""
    if not camera_info:
        return

    print("=" * 60)
    print("生成配置文件模板")
    print("=" * 60)
    print()

    # 生成 JSON 配置
    print("将以下内容保存为 cameras/camera_config.json:\n")
    print("{")
    print('  "cameras": [')

    view_names = ['front_view', 'side_view', 'top_view', 'wrist_view']

    for i, cam in enumerate(camera_info):
        view_name = view_names[i] if i < len(view_names) else f'camera_{i}'

        print("    {")
        print(f'      "name": "{view_name}",')
        print('      "type": "realsense",')
        print(f'      "serial_number": "{cam["serial_number"]}",')
        print('      "width": 1280,')
        print('      "height": 720,')
        print('      "fps": 30,')
        print('      "enable_depth": true,')
        print('      "enable_color": true,')
        print('      "align_depth_to_color": true')

        if i < len(camera_info) - 1:
            print("    },")
        else:
            print("    }")

    print("  ]")
    print("}")
    print()

    # # 生成 Python 代码示例
    # print("=" * 60)
    # print("Python 代码示例")
    # print("=" * 60)
    # print()
    # print("from cameras import CameraManager")
    # print()
    # print("# 方法1: 从配置文件加载")
    # print('camera_manager = CameraManager(config_path="cameras/camera_config.json")')
    # print()
    # print("# 方法2: 直接传入配置")
    # print("cameras_config = [")

    # for i, cam in enumerate(camera_info):
    #     view_name = view_names[i] if i < len(view_names) else f'camera_{i}'
    #     print("    {")
    #     print(f"        'name': '{view_name}',")
    #     print("        'type': 'realsense',")
    #     print(f"        'serial_number': '{cam['serial_number']}',")
    #     print("        'width': 1280,")
    #     print("        'height': 720,")
    #     print("        'fps': 30,")
    #     print("        'enable_depth': True")

    #     if i < len(camera_info) - 1:
    #         print("    },")
    #     else:
    #         print("    }")

    # print("]")
    # print("camera_manager = CameraManager(cameras_config=cameras_config)")
    # print()
    # print("# 启动所有相机")
    # print("camera_manager.start_all()")
    # print()
    # print("# 读取所有相机的帧")
    # print("frames = camera_manager.read_all_frames()")
    # print("for cam_name, frame_data in frames.items():")
    # print("    print(f'{cam_name}: {frame_data[\"color\"].shape}')")
    # print()
    # print("# 停止所有相机")
    # print("camera_manager.stop_all()")
    # print()


def main():
    """主函数"""
    print()
    camera_info = list_realsense_cameras()

    if camera_info:
        print()
        generate_config(camera_info)

        # print("=" * 60)
        # print("提示")
        # print("=" * 60)
        # print()
        # print("1. 复制上面的 JSON 配置到 cameras/camera_config.json")
        # print("2. 根据实际情况修改 view 名称 (front_view, side_view 等)")
        # print("3. 调整分辨率和帧率参数")
        # print("4. 运行数据采集:")
        # print("   python franka_data_collector.py --camera-config cameras/camera_config.json")
        # print()


if __name__ == '__main__':
    main()
