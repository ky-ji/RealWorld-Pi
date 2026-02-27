#!/bin/bash
# GR00T 代码同步脚本
# 用于将代码同步到远程服务器

# 配置
source sync_config.sh

# 同步代码
echo "同步代码到 $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='wandb' \
    --exclude='*.egg-info' \
    --exclude='log/' \
    -e "ssh -i $SSH_KEY -p $SSH_PORT" \
    "$LOCAL_PATH/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/"

echo "✓ 代码同步完成"

