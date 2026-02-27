# 数据同步

将采集的数据同步到训练服务器。

## 使用方法

```bash
# 同步数据集
rsync -avz --progress \
    -e "ssh -i ~/.ssh/id_rsa" \
    /path/to/local/data/ \
    user@server:/path/to/remote/data/
```

## 数据格式

参考 GR00T 数据准备文档：[data_preparation.md](../../../getting_started/data_preparation.md)

