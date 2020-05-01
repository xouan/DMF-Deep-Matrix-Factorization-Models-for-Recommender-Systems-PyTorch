#!/usr/bin/env bash


dataset_name='ml-1m'
python GMF.py --dataset  $dataset_name --epochs 200 --batch_size 512 --reg 0.0000 --num_factors 128  --num_neg 1 --lr 0.0005