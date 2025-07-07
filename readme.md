# 狐呼标

**[中文文档](readme-zh.md)**

**Now You can label "AP" and "SP" with HTK label file! PLEASE use `htk_lab_add_ap.py` to label it!**

## Intro

Use a neural network model to annotate the breathing (AP) in the textgrid file.

Note:

1. Only label "AP" in the "SP" or "" label of the original tg file, and accuracy is based on the original annotation
   file.

2. Only supports tg files with two layers of annotations: the words and the phones.

[Cpp Version (Beta)](https://github.com/openvpi/dataset-tools/releases/)

The CPP version has a UI interface, but cannot be accelerated using a graphics card.

## How to use

0. If using [SOFA](https://github.com/qiuqiao/SOFA) to generate textgrid annotations
    ```bash
    python infer.py ... --ap_detector NoneAPDetector
    ```
   An additional "--ap_detector NoneAPDetector" needs to be added to generate a tg file without AP annotations.

1. Download model

   [Model Link](https://github.com/autumn-DL/FoxBreatheLabeler/releases/latest)

   model_folder

   ├── config.yaml

   └── model_ckpt_steps_7000.ckpt

2. Generate AP labels by running textgrid-add-ap
   ```bash
   python textgrid_add_ap.py --ckpt_path model_folder/xx.ckpt --wav_dir wav_dir --tg_dir tg_dir --tg_out_dir tg_out_dir
   
   Option:
       --ckpt_path     str    Path to the checkpoint
       --wav_dir       str    Wav file folder (*.wav).
       --tg_dir        str    Textgrid files (*.TextGrid).
       --tg_out_dir    str    Output path of tg file after labeling AP.
       --ap_threshold  float  default: 0.4   Respiratory probability recognition threshold.  (Option)
       --ap_dur        float  default: 0.08  The shortest duration of breathing, discarded below this threshold, in seconds. (Option)
       --sp_dur        float  default: 0.1   SP fragments below this threshold will be adsorbed onto adjacent AP, in seconds.   (Option)
   ```

## ReLabel

ReLabel the TG file with breathing.

1. Clear AP in original label.

   ```bash
   python clean_ap.py --tg_dir raw_tg_dir --clean_tg_dir clean_tg_dir
   
    Option:
        --tg_dir        str    Textgrid files (*.TextGrid).
        --clean_tg_dir  str    Clean textgrid output dir (*.TextGrid).
        --phonemes      str    default: AP,SP,  The phonemes to be cleared are separated by English commas.  (Option)
   ```

2. Generate AP labels by running textgrid-add-ap(to clean_tg_dir)

## About `htk_lab_add_ap.py`

You can use "SP" or "pau" in HTK label file with it.

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
