# 代码同步工具

快速将代码同步到真机主机或推理服务器。

## 🚀 快速开始

### 1. 配置目标地址

编辑 `sync_config.sh`，设置你的目标地址和密码：

```bash
# 真机主机配置
ROBOT_HOST="192.168.1.100"       # 修改为你的真机主机地址
ROBOT_USER="user"                 # 修改为你的用户名
ROBOT_PORT=22                     # SSH 端口
ROBOT_PASSWORD="your_password"    # SSH 密码（留空将提示输入）
ROBOT_TARGET_DIR="/home/user/robot_inference"  # 修改为目标目录

# 推理服务器配置（如需修改）
SERVER_HOST="115.190.134.186"
SERVER_USER="jikangye"
SERVER_PORT=22
SERVER_PASSWORD="your_password"   # SSH 密码（留空将提示输入）
```

**注意**: 
- 如果配置了密码，建议安装 `sshpass` 以实现自动登录：`sudo apt-get install sshpass` (Ubuntu/Debian)
- 如果未配置密码或未安装 `sshpass`，脚本会提示你手动输入密码

### 2. 添加执行权限

```bash
chmod +x sync.sh sync_config.sh
```

### 3. 同步代码

```bash
# 同步到真机主机
./sync.sh robot

# 同步到推理服务器
./sync.sh server

# 同步到所有目标
./sync.sh all
```

## 📋 命令参考

| 命令 | 说明 |
|------|------|
| `./sync.sh robot` | 同步 robot_inference 到真机主机 |
| `./sync.sh server` | 同步 server 代码到推理服务器 |
| `./sync.sh all` | 同步到所有目标 |
| `./sync.sh robot -d` | 预览模式（不实际传输） |
| `./sync.sh -c` | 显示当前配置 |
| `./sync.sh -h` | 显示帮助 |

## 📁 同步内容

### 同步到真机主机 (`robot`)

将 `robot_inference/` 目录同步到真机，包含：
- `inference_client.py` - SSH 隧道推理客户端
- `inference_client_local.py` - 本地推理客户端
- `configs/` - 配置文件
- `control/` - 控制代码和相机模块
- `keys/` - SSH 密钥

### 同步到推理服务器 (`server`)

将整个 `realworld_deploy/` 目录同步到服务器，包含：
- `server/` - 推理服务器代码
- `robot_inference/` - 客户端代码（用于测试）

## ⚙️ 高级用法

### 指定目标目录

```bash
# 同步到自定义目录
./sync.sh robot --robot-dir /tmp/my_robot_code
./sync.sh server --server-dir /home/user/my_project
```

### 预览同步（干跑模式）

```bash
# 查看将要同步的文件，不实际执行
./sync.sh robot -d
```

## 🔧 配置说明

### sync_config.sh 配置项

```bash
# 真机主机
ROBOT_HOST="IP地址"
ROBOT_USER="用户名"
ROBOT_PORT=22
ROBOT_PASSWORD=""                 # SSH 密码（留空将提示输入）
ROBOT_TARGET_DIR="目标目录"

# 推理服务器
SERVER_HOST="IP地址"
SERVER_USER="用户名"
SERVER_PORT=22
SERVER_PASSWORD=""                # SSH 密码（留空将提示输入）
SERVER_TARGET_DIR="目标目录"
```

### 排除文件

以下文件/目录会被自动排除：
- `__pycache__/`
- `*.pyc`, `*.pyo`, `*.pyd`
- `.git/`
- `*.log`, `log/`, `logs/`
- `*.ckpt`, `*.pth`
- `data/`
- `.ipynb_checkpoints/`

## 💡 使用流程

### 日常开发流程

1. 在本地修改代码
2. 运行 `./sync.sh robot` 同步到真机
3. SSH 到真机运行测试

### 部署服务器

1. 修改代码
2. 运行 `./sync.sh server` 同步服务器代码
3. SSH 到服务器启动推理服务

## ❓ 常见问题

### SSH 连接失败

1. 检查 IP 地址和端口是否正确
2. 检查密码是否正确（如果使用密码登录）
3. 检查网络连接
4. 如果使用 `sshpass`，确保已安装：`sudo apt-get install sshpass`

### 权限不足

```bash
chmod +x sync.sh sync_config.sh
```

### 同步很慢

- 确保网络连接稳定
- 使用预览模式 `-d` 检查是否同步了不必要的大文件

