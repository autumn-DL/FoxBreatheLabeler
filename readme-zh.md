# 狐呼标

## 介绍

使用ASR(语音识别)获得音节, 同歌词中的文本进行匹配, 并生成可供Minlabel加载的JSON.

注意: 仅在原始textgrid中的"SP"中标记"AP", 准确性取决于原始标注文件

[Cpp 版本 (测试)](https://github.com/openvpi/dataset-tools/releases/tag/20240617.01)

CPP版本拥有ui界面,提供了量化过后体积更小的模型, 但不能使用显卡加速

## 如何使用

0. 使用[SOFA](https://github.com/qiuqiao/SOFA) 生成textgrid标注
    ```bash
    python infer.py ... --ap_detector NoneAPDetector
    ```
   需要添加"--ap_detector NoneAPDetector"选项以生成没有AP标注的textgrid文件
        
1. 下载模型

   [模型链接](https://github.com/autumn-DL/FoxBreatheLabeler/releases/latest)
   
   model_folder

     ├── config.yaml

     └── model_ckpt_steps_7000.ckpt

2.  通过运行textgrid_add_ap生成AP标注
    ```bash
    python textgrid_add_ap.py --ckpt_path model_folder/xx.ckpt --wav_dir wav_dir --tg_dir tg_dir --tg_out_dir tg_out_dir
    
    选项:
        --ckpt_path     str    模型路径
        --wav_dir       str    Wav音频文件目录 (*.wav).
        --tg_dir        str    Textgrid目录 (*.TextGrid).
        --tg_out_dir    str    输出文件夹
        --ap_threshold  float  默认: 0.4   呼吸识别阈值  (可选)
        --ap_dur        float  默认: 0.08  最短的呼吸时间，低于此阈值的会被舍弃, 以秒为单位. (可选)
        --sp_dur        float  默认: 0.1   低于此阈值的SP将被吸附到临近的AP上, 以秒为单位.   (可选)
    ```