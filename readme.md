# 狐呼标

## Intro

Using ASR to obtain syllables, matching text from lyrics, and generating JSON for Minlabel preloading.

Note: Only label "AP" in the "SP" label of the original tg file, and accuracy is based on the original annotation file.

[Cpp Version (Beta)](https://github.com/openvpi/dataset-tools/releases/tag/20240617.01)

The CPP version has a UI interface that provides a smaller volume for downloading quantization models, but cannot be accelerated using a graphics card.

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

2.  Generate AP labels by running textgrid-add-ap
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