# 基于Emotion2Vec模型的批量音频情感自动标注工具
## 简介
基于[emotion2vec](https://www.modelscope.cn/models/iic/emotion2vec_base_finetuned/summary)与[StructBERT](https://www.modelscope.cn/models/iic/nlp_structbert_emotion-classification_chinese-base/summary)对输入的音频进行情绪八分类（生气、厌恶、恐惧、开心、中立、其他、难过、吃惊）

此项目包含一个音频时长筛选/批量重命名脚本`preprocess_audio.py`批量推理脚本`recognize.py`，和一个音频分类脚本`classify.py`和一个`webui.py`界面

## 依赖项

- Python 3.10
- pydub
- modelscope
- funasr
- torchaudio
- Gardio
- 安装所需依赖`pip install pydub modelscope funasr torchaudio Gardio

#快速使用

```
input
└───speaker
   ├───xxx.wav
   └─── subtitle.lab
```
数据集格式可以参考此[数据集](https://github.com/AI-Hobbyist/Genshin_Datasets)，或者在音频名上标注音频内容否则无法使用文本情感识别功能。
参考输出为此格式
```
output
└───speaker0
   └───emotion
       └───【emotion】{text}      
```
经过处理后的文件将会被自动归类并重命名，以反映出音频中包含的情感及其相关内容。一个典型的输出文件示例如下：
    `output/小明/生气/【生气】我很生气.wav`
    `output/小明/开心/【开心】我很开心.wav`

