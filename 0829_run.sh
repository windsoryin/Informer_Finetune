7_8_run.sh#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=18

yhrun -N 1 -p gpu --gpus-per-node=1 --cpus-per-gpu=8 python main_informer.py --model informer --data JFNG_data_15min --data_path JFNG_data_15min.csv --seq_len 48 --label_len 24 --pred_len 12

