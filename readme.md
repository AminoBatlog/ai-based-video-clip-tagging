# 基于Sensevoice的AI辅助视频高光时刻检测工具

## 使用流程

1. 安装python环境

```
pip install -r requirements.txt
```
安装完成后，需手动下载pytorch

若为英伟达显卡用户，根据显卡类型选择对应的cuda版本安装对应的pytorch https://pytorch.org/get-started/locally/

如CUDA版本为12.9，则下载代码为
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129
```

如果不下载支持对应cuda版本的pytorch，则会使用cpu进行AI模型的运行，会更慢一些

AMD或Intel显卡用户则只能使用cpu进行AI模型运行

下载cpu版本的pytorch代码为
```
pip3 install torch torchvision
```


2. 修改参数
根据自行需要调整代码31到38行的参数
```python
# 需要检测的关键词列表
TARGET_WORDS = ["牛", "强", "厉害", "卧槽", "准", "锁", "傻逼"]
# SenseVoice 中可能输出的事件标签
EVENT_TAGS = {
    "laughter": ["<|Laughter|>", "[笑声]", "laughter"]
}
TARGET_EVENTS = ["laughter"]
target_folder = "Z:/test/"  # 替换为视频目录
```
根据需要调整TARGET_WORDS和target_folder参数
3. 运行代码
