#!/bin/bash


if [ $# -lt 2 ]; then
  echo "use: $0 <gpu_id> <task_config_path> [save_dir]"
  exit 1
fi

GPU_ID=$1
TASK_CONFIG=$2
SAVE_DIR=${3:-'./lsun_results'}


CUDA_VISIBLE_DEVICES=$GPU_ID python3 main.py \
  --model_config=configs/model_config_lsunbedroom.yaml \
  --diffusion_config=configs/diffusion_config.yaml \
  --task_config=$TASK_CONFIG \
  --save_dir=$SAVE_DIR
