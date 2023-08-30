#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=18

yhrun -N 1 -p gpu1 --gpus-per-node=1 --cpus-per-gpu=8 python main_informer.py --model informer --data JFNG_data_15min_1year --data_path JFNG_data_15min_1year.csv