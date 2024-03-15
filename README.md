# 基于Emotion2Vec模型的批量音频情感自动标注工具
## 简介
基于[emotion2vec](https://www.modelscope.cn/models/iic/emotion2vec_base_finetuned/summary)对输入的音频进行情绪八分类（生气、厌恶、恐惧、开心、中立、其他、难过、吃惊）

此项目包含一个批量推理脚本`recognize.py`，和一个音频分类脚本`classify.py`

## 部署
安装依赖

`pip install modelscope torchaudio pydub`
