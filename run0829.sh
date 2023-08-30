#!/bin/bash

python -u main_informer.py --data JFNG_data_15min --seq_len 48 --label_len 24 --pred_len 4 --itr 1
#python -u main_informer.py --seq_len 96 --label_len 48 --pred_len 24 --itr 1
#python -u main_informer.py --seq_len 96 --label_len 48 --pred_len 12 --itr 1
#python -u main_informer.py --seq_len 168 --label_len 96 --pred_len 96 --itr 1
#python -u main_informer.py --seq_len 168 --label_len 96 --pred_len 48 --itr 1
#python -u main_informer.py --seq_len 168 --label_len 96 --pred_len 24 --itr 1
#python -u main_informer.py --seq_len 168 --label_len 96 --pred_len 12 --itr 1