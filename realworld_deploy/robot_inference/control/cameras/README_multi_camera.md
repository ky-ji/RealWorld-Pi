# 多相机数据采集功能说明

## 功能概述

本系统现已支持多相机数据采集，可以同时使用多个 RealSense 或 USB 相机从不同视角采集图像和深度数据。

## 主要特性

- ✅ 支持多个 Intel RealSense D435i 相机（通过序列号区分）
- ✅ 支持多个 USB 相机（通过索引区分）
- ✅ 支持混合使用不同类型的相机
- ✅ 使用描述性名称标识每个相机视角（如 `front_view`, `side_view`）
- ✅ 并行读取所有相机帧，提高采集效率
- ✅ 为每个相机视角创建独立的图像和深度文件夹
- ✅ 完全向后兼容单相机模式

## 文件结构

### 多相机模式下的数据存储结构

```
data/trajectories/
└── episode_0001/
    ├── data.pkl                    # 轨迹数据（包含所有相机的索引）
    ├── meta.json                   # 元数据
    ├── Images_front_view/          # 前视相机图像
    │   ├── frame_0000.jpg
    │   ├── frame_0001.jpg
    │   └── ...
    ├── Depth_front_view/           # 前视相机深度
    │   ├── depth_0000.npy
    │   ├── depth_0001.npy
    │   └── ...
    ├── Images_side_view/           # 侧视相机图像
    │   ├── frame_0000.jpg
    │   ├── frame_0001.jpg
    │   └── ...
    └── Depth_side_view/            # 侧视相机深度
        ├── depth_0000.npy
        ├── depth_0001.npy
        └── ...
```

### 单相机模式（兼容旧代码）

```
data/trajectories/
└── episode_0001/
    ├── data.pkl
    ├── meta.json
    ├── Images/
    │   └── frame_*.jpg
    └── Depth/
        └── depth_*.npy
```

## 配置文件格式

在 `cameras/` 文件夹中创建一个 JSON 配置文件来定义多个相机，例如 `cameras/camera_config.json`：

```json
{
  "cameras": [
    {
      "name": "front_view",
      "type": "realsense",
      "serial_number": "123456789",
      "width": 1280,
      "height": 720,
      "fps": 30,
      "enable_depth": true,
      "enable_color": true,
      "align_depth_to_color": true
    },
    {
      "name": "side_view",
      "type": "realsense",
      "serial_number": "987654321",
      "width": 1280,
      "height": 720,
      "fps": 30,
      "enable_depth": true,
      "enable_color": true,
      "align_depth_to_color": true
    }
  ]
}
```

### 配置参数说明

每个相机配置项包含以下参数：

- **name** (必需): 相机视角的描述性名称，如 `front_view`, `side_view`, `wrist_view` 等
- **type** (必需): 相机类型，支持 `realsense` 或 `usb`
- **serial_number** (RealSense 必需): RealSense 相机的序列号，用于区分多个相机
- **camera_index** (USB 相机必需): USB 相机的索引号（0, 1, 2...）
- **width**: 图像宽度（默认 640）
- **height**: 图像高度（默认 480）
- **fps**: 帧率（默认 30）
- **enable_depth** (RealSense): 是否启用深度采集（默认 true）
- **enable_color** (RealSense): 是否启用彩色图像采集（默认 true）
- **align_depth_to_color** (RealSense): 是否将深度对齐到彩色图像（默认 true）

## 使用方法

### 1. 查找 RealSense 相机序列号

我们提供了一个便捷的工具脚本来查找所有连接的 RealSense 相机并自动生成配置文件：

```bash
# 运行工具脚本
python cameras/list_cameras.py
```

这个脚本会：
- 列出所有连接的 RealSense 相机及其序列号
- 自动生成配置文件模板
- 提供 Python 代码示例

输出示例：
```
============================================================
查找 RealSense 相机
============================================================

✓ 找到 2 个 RealSense 设备:

设备 1:
  名称:     Intel RealSense D435I
  序列号:   123456789
  固件版本: 5.12.7.100
  USB 类型: 3.2

设备 2:
  名称:     Intel RealSense D435I
  序列号:   987654321
  固件版本: 5.12.7.100
  USB 类型: 3.2
```

### 2. 创建配置文件

将工具脚本生成的 JSON 配置保存到 `cameras/camera_config.json`。

### 3. 启动数据采集

使用 `--camera-config` 参数指定配置文件：

```bash
# 多相机模式
python franka_data_collector.py --camera-config cameras/camera_config.json

# 单相机模式（兼容旧代码）
python franka_data_collector.py --camera-type realsense --camera-width 1280 --camera-height 720
```

### 4. 数据加载示例

```python
import pickle
import numpy as np
from pathlib import Path

# 加载 episode 数据
episode_path = Path("data/trajectories/episode_0001")
with open(episode_path / "data.pkl", "rb") as f:
    data = pickle.load(f)

# 检查是否为多相机模式
if data.get('multi_camera_mode', False):
    print(f"多相机模式，相机列表: {data['camera_names']}")

    # 访问每个相机的图像索引
    for cam_name in data['camera_names']:
        image_indices = data['image_indices'][cam_name]
        depth_indices = data['depth_indices'][cam_name]
        print(f"\n{cam_name}:")
        print(f"  图像数: {len(image_indices)}")
        print(f"  深度数: {len(depth_indices)}")

        # 加载第一帧图像
        if image_indices[0] >= 0:
            import cv2
            img_path = episode_path / f"Images_{cam_name}" / f"frame_{image_indices[0]:04d}.jpg"
            image = cv2.imread(str(img_path))
            print(f"  图像尺寸: {image.shape}")

        # 加载第一帧深度
        if depth_indices[0] >= 0:
            depth_path = episode_path / f"Depth_{cam_name}" / f"depth_{depth_indices[0]:04d}.npy"
            depth = np.load(str(depth_path))
            print(f"  深度尺寸: {depth.shape}")
else:
    print("单相机模式")
    image_indices = data['image_index']
    depth_indices = data['depth_index']
```

## API 使用示例

### 使用 CameraManager

```python
from cameras import CameraManager

# 从配置文件创建
camera_manager = CameraManager(config_path="cameras/camera_config.json")

# 或直接传入配置
cameras_config = [
    {
        'name': 'front_view',
        'type': 'realsense',
        'serial_number': '123456789',
        'width': 1280,
        'height': 720,
        'fps': 30,
        'enable_depth': True
    },
    {
        'name': 'side_view',
        'type': 'realsense',
        'serial_number': '987654321',
        'width': 1280,
        'height': 720,
        'fps': 30,
        'enable_depth': True
    }
]
camera_manager = CameraManager(cameras_config=cameras_config)

# 启动所有相机
camera_manager.start_all()

# 读取所有相机的帧（并行）
frames = camera_manager.read_all_frames(parallel=True)
for cam_name, frame_data in frames.items():
    color = frame_data['color']
    depth = frame_data['depth']
    timestamp = frame_data['timestamp']
    print(f"{cam_name}: color={color.shape if color is not None else None}")

# 停止所有相机
camera_manager.stop_all()

# 或使用上下文管理器
with CameraManager(config_path="cameras/camera_config.json") as cam_mgr:
    frames = cam_mgr.read_all_frames()
    # 处理帧...
```

## 性能优化

- **并行读取**: 多相机模式默认使用并行读取，可显著提高采集效率
- **缓冲区管理**: 自动清空缓冲区确保时间同步
- **线程池**: 使用 `ThreadPoolExecutor` 并发读取所有相机

## 注意事项

1. **USB 带宽**: 多个 RealSense 相机需要足够的 USB 带宽，建议：
   - 使用 USB 3.0 或更高版本
   - 每个相机连接到不同的 USB 控制器
   - 降低分辨率或帧率以减少带宽需求

2. **序列号**: 确保配置文件中的序列号与实际相机匹配

3. **存储空间**: 多相机会产生更多数据，确保有足够的存储空间

4. **向后兼容**: 不提供 `--camera-config` 参数时，系统自动使用单相机模式

## 故障排除

### 问题：找不到相机序列号

```bash
# 方法1: 使用我们提供的工具脚本（推荐）
python cameras/list_cameras.py

# 方法2: 使用 RealSense SDK 命令行工具
rs-enumerate-devices
```

### 问题：相机启动失败

- 检查 USB 连接
- 确认序列号正确
- 尝试降低分辨率或帧率
- 检查是否有其他程序占用相机

### 问题：数据采集速度慢

- 启用并行读取（默认已启用）
- 降低图像分辨率
- 减少相机数量
- 禁用不需要的深度采集

## 示例配置

### 双 RealSense 相机（高分辨率）

```json
{
  "cameras": [
    {
      "name": "front_view",
      "type": "realsense",
      "serial_number": "YOUR_SERIAL_1",
      "width": 1920,
      "height": 1080,
      "fps": 30,
      "enable_depth": true
    },
    {
      "name": "side_view",
      "type": "realsense",
      "serial_number": "YOUR_SERIAL_2",
      "width": 1920,
      "height": 1080,
      "fps": 30,
      "enable_depth": true
    }
  ]
}
```

### 混合相机配置

```json
{
  "cameras": [
    {
      "name": "wrist_view",
      "type": "realsense",
      "serial_number": "YOUR_SERIAL",
      "width": 640,
      "height": 480,
      "fps": 30,
      "enable_depth": true
    },
    {
      "name": "overhead_view",
      "type": "usb",
      "camera_index": 0,
      "width": 1280,
      "height": 720,
      "fps": 30
    }
  ]
}
```

## 更新日志

- **2025-12-23**: 添加多相机支持
  - 新增 `CameraManager` 类
  - 支持 JSON 配置文件
  - 并行读取优化
  - 完全向后兼容单相机模式
