#!/usr/bin/env bash

# To see the performance of your trained AI
# sh ./watch_ai.sh
# then you will see player1 in red and player2 in blue

# method list - {'smv1', 'smv2', 'nfsp', 'fpemv1', 'fpemv2'}

python3 ./script/visiable.py --p1_method fpemv1 --p1_path ./model/fpemv1/  --p2_method fpemv2 --p2_path ./model/fpemv2