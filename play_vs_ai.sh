#!/usr/bin/env bash

# Run this if you want to play with the AI you trained before
# sh ./play_with_ai.py
MODEL_PATH=$1

python ./script/play_vs_ai.py --method fpemv1 --load ${MODEL_PATH}