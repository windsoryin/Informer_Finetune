#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=18

yhrun -N 1 -p gpu1 --gpus-per-node=4 --cpus-per-gpu=8 python test02.py --use_multi_gpu
