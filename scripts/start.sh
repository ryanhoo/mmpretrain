#!/bin/bash
# mim install --no-cache-dir -e .
# pip install torch-model-archiver

# python tools/train.py configs/custom/custom.py

# python tools/train.py configs/custom/efficientnetv2_s.py
# ./demo/ipu_train_example.sh
cfg_name=efficientnetv2_s_3nd # custom
cfg_file=configs/custom/${cfg_name}.py
checkpoint_file=work_dirs/${cfg_name}/epoch_62.pth

# train
# python3 tools/train.py configs/custom/${cfg_name}.py &&
# python3 tools/test.py ${cfg_file} ${checkpoint_file} # --metrics accuracy --device ipu


# distrubute
image_name=zbbl-junior.jpg
python demo/image_demo.py demo/${image_name} ${cfg_file} --checkpoint ${checkpoint_file}
