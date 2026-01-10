import fsspec
import os
import shutil
import pathlib

# 定义源路径和目标路径
source_url = "gs://openpi-assets/checkpoints/pi05_base/params"
dest_path = "/data3/yinmenghao/code/openpi/local_model/pi05_base_lora/params"

# 确保目标路径不存在
dest_path_obj = pathlib.Path(dest_path)
if dest_path_obj.exists():
    shutil.rmtree(dest_path_obj)
dest_path_obj.mkdir(parents=True, exist_ok=True)

print(f"正在从 {source_url} 下载模型到 {dest_path}")

# 使用 fsspec 下载整个目录
fs, _ = fsspec.core.url_to_fs(source_url)
fs.get(source_url, dest_path, recursive=True)

print(f"模型下载完成!")
print(f"下载的模型大小: {shutil.disk_usage(dest_path).used / 1024 / 1024:.2f} MB")
