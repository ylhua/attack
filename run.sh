#!/bin/bash
#
# run_attack.sh is a script which executes the attack
#
# Envoronment which runs attacks and defences calls it in a following way:
#   run_attack.sh INPUT_DIR OUTPUT_DIR MAX_EPSILON
# where:
#   INPUT_DIR - directory with input PNG images
#   OUTPUT_DIR - directory where adversarial images should be written
#   MAX_EPSILON - maximum allowed L_{\infty} norm of adversarial perturbation
#
# -*- coding: utf-8 -*-
INPUT_DIR=$1
OUTPUT_DIR=$2
MAX_EPSILON=$3

python attack_iter.py \
  --input_dir="/home/huahua/PycharmProjects/Dehaze-GAN-master/1/" \
  --output_dir="./output_image" \
  --max_epsilon='16' \
  --checkpoint_path_inception_v1=./models/inception_v1/inception_v1.ckpt \
  --checkpoint_path_resnet=./models/resnet_v1_50/model.ckpt-49800 \
  --checkpoint_path_vgg=./models/vgg_16/vgg_16.ckpt \
  --num_iter=10 \
  --momentum=1.0
