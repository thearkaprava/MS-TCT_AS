#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python train.py \
-dataset tsu \
-mode rgb \
-model MS_TCT \
-train True \
-rgb_root /path/to/smarthome_features_i3d/ \
-num_clips 2500 \
-skip 0 \
-lr 0.0001 \
-comp_info False \
-epoch 140 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 1 \
-output_dir workdirs/output/