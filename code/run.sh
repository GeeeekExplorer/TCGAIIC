#!/bin/bash
python preprocess.py
for i in 0 1 2 3
do { sleep $i; python pretrain.py & } done
wait
cp -r ../user_data/pretrain/0 ../user_data/pretrain/4
cp -r ../user_data/pretrain/1 ../user_data/pretrain/5
for i in 0 1 2 3
do { sleep $i; python train.py & } done
wait
for i in 4 5
do { sleep $i; python train.py & } done
wait
python infer.py
#python infer.py &
#sleep 30
#python request.py
