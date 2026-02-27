#!/bin/bash
# 同步配置

# 远程服务器配置
REMOTE_HOST="your-gpu-server.com"
REMOTE_USER="username"
REMOTE_PATH="/path/to/Isaac-GR00T"
SSH_PORT=22
SSH_KEY="$HOME/.ssh/id_rsa"

# 本地路径
LOCAL_PATH="$(cd "$(dirname "$0")/../.." && pwd)"

