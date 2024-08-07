# 狐呼标

## 介绍

使用神经网络模型标注出textgrid文件中的呼吸（AP）。

注意:

1. 仅在原始textgrid中的"SP"或""中标记"AP", 准确性取决于原始标注文件。

2. 仅支持两层标注的tg文件：“words”、“phones”。

[Cpp 版本 (测试)](https://github.com/openvpi/dataset-tools/releases/)

CPP版本拥有ui界面, 但不能使用显卡加速。

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

2. 通过运行textgrid_add_ap生成AP标注
   ```bash
   python textgrid_add_ap.py --ckpt_path model_folder/xx.ckpt --wav_dir wav_dir --tg_dir tg_dir --tg_out_dir tg_out_dir
   
   选项:
       --ckpt_path     str    模型路径
       --wav_dir       str    Wav音频文件目录 (*.wav).
       --tg_dir        str    Textgrid目录 (*.TextGrid).
       --tg_out_dir    str    输出文件夹
       --ap_threshold  float  默认: 0.4   呼吸识别阈值  (可选)
       --ap_dur        float  默认: 0.08  最短的呼吸时间，低于此阈值的会被舍弃, 以秒为单位. (可选)
       --sp_dur        float  默认: 0.1   低于此阈值的SP将被吸附到临近的AP上, 以秒为单位.  (可选)
   ```

## 重新标注

重新标注已包含呼吸（AP）的tg文件。

1. 清除原标注中的呼吸.

   ```bash
   python clean_ap.py --tg_dir raw_tg_dir --clean_tg_dir clean_tg_dir
   
    选项:
        --tg_dir        str    Textgrid目录 (*.TextGrid).
        --clean_tg_dir  str    清空AP、SP的Textgrid目录 (*.TextGrid).
        --phonemes      str    default: AP,SP,  待清空的音素，以英文逗号分隔.  (可选)
   ```

2. 通过运行textgrid_add_ap生成AP标注(对clean_tg_dir)