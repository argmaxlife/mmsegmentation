#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/substation/oilleak0.py 2 --work-dir ../../experiments/oilleak/20250706_v7.2_swin_bs8_e100_kfold/0
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/substation/oilleak1.py 2 --work-dir ../../experiments/oilleak/20250706_v7.2_swin_bs8_e100_kfold/1
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/substation/oilleak2.py 2 --work-dir ../../experiments/oilleak/20250706_v7.2_swin_bs8_e100_kfold/2
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/substation/oilleak3.py 2 --work-dir ../../experiments/oilleak/20250706_v7.2_swin_bs8_e100_kfold/3
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/substation/oilleak4.py 2 --work-dir ../../experiments/oilleak/20250706_v7.2_swin_bs8_e100_kfold/4