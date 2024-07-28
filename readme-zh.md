# 狐呼标

**现在可以使用`htk_lab_add_ap.py`来标注出HTK label中的"AP"和"SP"了**

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

## 重新标注（新增HTK lab支持）

重新标注已包含呼吸（AP）的TextGrid/HTK lab文件。

1. 清除原标注中的呼吸.

   ```bash
   python clean_ap.py --label_dir raw_label_dir --clean_label_dir clean_label_dir
   
    选项:
        --label_dir        str    标注目录 (TextGrid or HTK lab).
        --clean_tg_dir  str    清空AP、SP的标注目录 (TextGrid or HTK lab).
        --phonemes      str    default: AP,SP,  待清空的音素，以英文逗号分隔.  (可选)
   ```

2. 通过运行textgrid_add_ap/htk_lab_add_ap生成AP标注(对clean_label_dir)

## 关于`htk_lab_add_ap.py`

使用此代码时，请保证HTK lab中包含"SP"或者"pau"。否则输出的HTK lab将与输入的HTK lab一致。

   ```bash
   python htk_lab_add_ap.py --ckpt_path model_folder/xx.ckpt --wav_dir wav_dir --lab_dir lab_dir --lab_out_dir lab_out_dir

   Options:
  --ckpt_path DIR       Path to the checkpoint  [required]
  --wav_dir DIR         Wav files  [required]
  --lab_dir DIR         Lab files  [required]
  --lab_out_dir DIR     Lab output dir  [required]
  --ap_threshold FLOAT  Respiratory probability recognition threshold
  --ap_dur FLOAT        The shortest duration of breathing, discarded below
                        this threshold, in seconds
  --sp_dur FLOAT        SP fragments below this threshold will adsorb to
                        adjacent AP, in seconds
  --help                Show this message and exit.

   ```
