#!/bin/bash

python main.py  --data_path_source ./data/datasets/Office31/  --src amazon  --data_path_target ./data/datasets/Office31/  --tar webcam_half  --data_path_target_t ./data/datasets/Office31/  --tar_t webcam_half2 #


python main.py  --data_path_source ./data/datasets/Office31/  --src amazon  --data_path_target ./data/datasets/Office31/  --tar dslr_half  --data_path_target_t ./data/datasets/Office31/  --tar_t dslr_half2 #


python main.py  --data_path_source ./data/datasets/Office31/  --src dslr  --data_path_target ./data/datasets/Office31/  --tar amazon_half  --data_path_target_t ./data/datasets/Office31/  --tar_t amazon_half2 #


python main.py  --data_path_source ./data/datasets/Office31/  --src webcam  --data_path_target ./data/datasets/Office31/  --tar amazon_half  --data_path_target_t ./data/datasets/Office31/  --tar_t amazon_half2 #


