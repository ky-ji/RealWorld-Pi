#!/usr/bin/env python3
"""
轻量级 GR00T PolicyClient

这是一个不依赖完整 gr00t 包的独立客户端实现。
只需要基础依赖: zmq, msgpack, numpy

用于解决 Python 版本冲突问题:
- gr00t 需要 Python >= 3.9 (因为 av, albumentations 等包)
- polymetis 需要 Python < 3.9

此客户端可以在 Python 3.8 的 polymetis 环境中运行，
通过 ZMQ 与运行在 Python 3.10 环境的 GR00T 服务器通信。
"""

import io
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import msgpack
import numpy as np
import zmq


class LightweightMsgSerializer:
    """轻量级消息序列化器"""
    
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=LightweightMsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=LightweightMsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        # 处理 ModalityConfig - 直接返回字典形式
        if "__ModalityConfig_class__" in obj:
            return obj["as_json"]
        # 处理 numpy 数组
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class LightweightPolicyClient:
    """
    轻量级策略客户端
    
    与 gr00t.policy.server_client.PolicyClient 功能兼容，
    但不需要导入完整的 gr00t 包。
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
        strict: bool = False,  # 保持与原 PolicyClient 接口兼容 (轻量级客户端忽略此参数)
    ):
        """
        初始化客户端
        
        Args:
            host: 服务器地址
            port: 服务器端口
            timeout_ms: 超时时间（毫秒）
            api_token: API 认证令牌
            strict: 是否严格检查 (忽略，仅用于接口兼容)
        """
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """初始化或重新初始化 socket"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")
        # 设置超时
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

    def ping(self) -> bool:
        """测试服务器连接"""
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # 重新创建 socket
            return False

    def kill_server(self):
        """关闭服务器"""
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: Optional[Dict] = None, requires_input: bool = True
    ) -> Any:
        """
        调用服务器端点
        
        Args:
            endpoint: 端点名称
            data: 请求数据
            requires_input: 是否需要输入数据
        
        Returns:
            服务器响应
        """
        request: Dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        if self.api_token:
            request["api_token"] = self.api_token

        self.socket.send(LightweightMsgSerializer.to_bytes(request))
        message = self.socket.recv()
        
        if message == b"ERROR":
            raise RuntimeError("Server error. Make sure the policy server is running.")
        
        response = LightweightMsgSerializer.from_bytes(message)

        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        
        return response

    def __del__(self):
        """清理资源"""
        try:
            self.socket.close()
            self.context.term()
        except:
            pass

    def get_action(
        self, observation: Dict[str, Any], options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        获取动作
        
        Args:
            observation: 观测字典
            options: 可选配置
        
        Returns:
            (action_dict, info_dict)
        """
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        return tuple(response)  # 将列表转为元组 (action, info)

    def reset(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """重置策略"""
        return self.call_endpoint("reset", {"options": options})

    def get_modality_config(self) -> Dict[str, Any]:
        """获取模态配置"""
        return self.call_endpoint("get_modality_config", requires_input=False)


# 为了兼容性，创建别名
PolicyClient = LightweightPolicyClient


if __name__ == "__main__":
    # 测试连接
    import argparse
    
    parser = argparse.ArgumentParser(description='测试轻量级 PolicyClient')
    parser.add_argument('--host', type=str, default='localhost', help='服务器地址')
    parser.add_argument('--port', type=int, default=5555, help='服务器端口')
    args = parser.parse_args()
    
    print(f"[测试] 连接到 {args.host}:{args.port}")
    
    client = LightweightPolicyClient(host=args.host, port=args.port)
    
    if client.ping():
        print("[测试] ✓ 连接成功!")
        
        try:
            config = client.get_modality_config()
            print(f"[测试] ✓ 模态配置: {list(config.keys()) if isinstance(config, dict) else config}")
        except Exception as e:
            print(f"[测试] ⚠ 获取配置失败: {e}")
    else:
        print("[测试] ✗ 连接失败")

