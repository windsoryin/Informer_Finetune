#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=18

yhrun -N 1 -p gpu1 --gpus-per-node=4 --cpus-per-gpu=8 python main_informer.py --use_multi_gpu --model informer --data JFNG_data_5min --data_path JFNG_data_5min.csv --seq_len 96 --label_len 96 --pred_len 24 --train_epochs 10 --batch_size 64 --freq 5t --d_model 16 --d_ff 64 --dropout 0.5 --learning_rate 0.001
yhrun -N 1 -p gpu1 --gpus-per-node=4 --cpus-per-gpu=8 python main_informer.py --use_multi_gpu --model informer --data JFNG_data_5min --data_path JFNG_data_5min.csv --seq_len 96 --label_len 48 --pred_len 72 --train_epochs 10 --batch_size 64 --freq 5t --d_model 16 --d_ff 64 --dropout 0.5 --learning_rate 0.001
yhrun -N 1 -p gpu1 --gpus-per-node=4 --cpus-per-gpu=8 python main_informer.py --use_multi_gpu --model informer --data JFNG_data_5min --data_path JFNG_data_5min.csv --seq_len 96 --label_len 48 --pred_len 24 --train_epochs 10 --batch_size 64 --freq 5t --d_model 16 --d_ff 64 --dropout 0.5 --learning_rate 0.001