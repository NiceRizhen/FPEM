#!/usr/bin/env bash

# Running time dependencies:
#   ubuntu
#   tensorflow: 1.4.0+
#   python 3.6

# run different method using
# sh ./run.sh [method_name]
# model will be saved at ./model/{method}/
# tensorboard log will be saved at ./log/{method}/

METHOD=$1

python ${METHOD}.py --process_num 16 --sample_epoch 125