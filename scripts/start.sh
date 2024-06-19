#!/bin/bash
# mim install --no-cache-dir -e .
# pip install torch-model-archiver

# python tools/train.py configs/custom/custom.py

# python tools/train.py configs/custom/efficientnetv2_s.py
# ./demo/ipu_train_example.sh
cfg_name=convnext_v2_n # custom
# cfg_name=efficientnetv2_s_3nd
cfg_file=configs/custom/${cfg_name}.py
checkpoint_file=work_dirs/${cfg_name}/epoch_100.pth

# train
python3 tools/train.py configs/custom/${cfg_name}.py &&
python3 tools/test.py ${cfg_file} ${checkpoint_file} # --metrics accuracy --device ipu


# distrubute
# image_name=zbbl-junior.jpg
# image_name=北红尾鸲雌.jpg
# image_name=htcwsq-snow.jpg
# image_name=cwfyy.jpg
# image_name=sunbird.jpg
# image_name=黑翅鸢.jpg
# image_name=zwhw.jpg
# python demo/image_demo.py demo/${image_name} ${cfg_file} --checkpoint ${checkpoint_file}
