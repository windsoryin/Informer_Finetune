#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=18

yhrun -N 1 -p gpu1 --gpus-per-node=4 --cpus-per-gpu=8 python main_informer.py --use_multi_gpu --model informer --data JFNG_data_15min --data_path JFNG_data_15min.csv
yhrun -N 1 -p gpu1 --gpus-per-node=4 --cpus-per-gpu=8 python main_informer.py --use_multi_gpu --model informer --data JFNG_data_15min_unrh --data_path JFNG_data_15min_unrh.csv
yhrun -N 1 -p gpu1 --gpus-per-node=4 --cpus-per-gpu=8 python main_informer.py --use_multi_gpu --model informer --data JFNG_data_15min_unsp --data_path JFNG_data_15min_unsp.csv
yhrun -N 1 -p gpu1 --gpus-per-node=4 --cpus-per-gpu=8 python main_informer.py --use_multi_gpu --model informer --data JFNG_data_15min_unt2m --data_path JFNG_data_15min_unt2m.csv
yhrun -N 1 -p gpu1 --gpus-per-node=4 --cpus-per-gpu=8 python main_informer.py --use_multi_gpu --model informer --data JFNG_data_15min_unwind --data_path JFNG_data_15min_unwind.csv
