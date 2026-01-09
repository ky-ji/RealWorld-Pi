from openpi.shared import download
import shutil
import os
import pathlib

# 使用 maybe_download 函数下载模型
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
print(f"JAX 模型已下载至: {checkpoint_dir}")

# 目标目录
local_model_dir = pathlib.Path("/data3/yinmenghao/code/openpi/local_model/pi05_base_lora")
local_model_dir.mkdir(parents=True, exist_ok=True)

# 将下载的模型复制到目标目录
print(f"正在将模型复制到: {local_model_dir}")

# 检查 checkpoint_dir 中的文件和目录
for item in os.listdir(checkpoint_dir):
    src_item = os.path.join(checkpoint_dir, item)
    
    # 如果是 params.partial 目录，复制其中的 params 子目录
    if item == "params.partial" and os.path.isdir(src_item):
        params_src = os.path.join(src_item, "params")
        params_dst = os.path.join(local_model_dir, "params")
        
        if os.path.exists(params_src):
            if os.path.exists(params_dst):
                shutil.rmtree(params_dst)
            shutil.copytree(params_src, params_dst)
            print(f"已复制目录: {item}/params -> params")
        else:
            print(f"警告: {item}/params 目录不存在")
    # 忽略 params.lock 文件
    elif item != "params.lock":
        dst_item = os.path.join(local_model_dir, item)
        if os.path.isdir(src_item):
            if os.path.exists(dst_item):
                shutil.rmtree(dst_item)
            shutil.copytree(src_item, dst_item)
            print(f"已复制目录: {item}")
        else:
            shutil.copy2(src_item, dst_item)
            print(f"已复制文件: {item}")

print("模型复制完成!")
