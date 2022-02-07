#!/bin/bash

CUDA_VISIBLE_DEVICES=0
max_len=512
batch_size=32

python -u main.py --max_len ${max_len} --batch_size ${batch_size} | tee log.txt