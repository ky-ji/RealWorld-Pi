import fsspec
import os
import pathlib
import shutil

# 定义源路径和目标路径
source_url = "gs://openpi-assets/checkpoints/pi05_base/assets"
dest_path = "/data3/yinmenghao/code/openpi/local_model/pi05_base_lora/assets"

# 确保目标路径存在
dest_path_obj = pathlib.Path(dest_path)
dest_path_obj.mkdir(parents=True, exist_ok=True)

print(f"正在从 {source_url} 下载assets到 {dest_path}")

# 使用 fsspec 直接下载整个目录
fs, _ = fsspec.core.url_to_fs(source_url)

# 检查源目录是否存在
if not fs.exists(source_url):
    print(f"源目录 {source_url} 不存在")
else:
    # 使用 fs.get 递归下载整个目录
    print(f"开始递归下载整个目录")
    fs.get(source_url, dest_path, recursive=True)
    print(f"递归下载完成")

print(f"assets下载完成!")
print(f"下载的文件列表:")
for root, _dirs, files in os.walk(dest_path):
    for file in files:
        print(f"  {os.path.join(root, file)}")

