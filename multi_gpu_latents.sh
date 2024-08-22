#!/bin/bash

TRAIN_DATA_DIR=/mnt/s3fs_cache/training_data_highres


TOTAL_SPLIT=32 # Total number of processes
OFFSET=0
CUDA_DEVICES_NUM=8 # Number of GPUs
PYTHON=python
MODEL_PATH=stabilityai/stable-diffusion-xl-base-1.0

# Launch 64 processes, 8 per GPU
for i in $(seq 0 $((TOTAL_SPLIT-1)))
do
    GPU_INDEX=$((i % CUDA_DEVICES_NUM))
    INDEX=$((i+OFFSET))
    CUDA_VISIBLE_DEVICES=$GPU_INDEX $PYTHON finetune/prepare_buckets_latents_separate.py \
        --split_dataset \
        --n_split $TOTAL_SPLIT \
        --current_index $INDEX \
        --model_name_or_path $MODEL_PATH \
        --max_resolution "1024,1024" \
        --min_bucket_reso=640 \
        --max_bucket_reso=1536 \
        --full_path \
        --skip_existing \
        --recursive \
        --train_data_dir $TRAIN_DATA_DIR &

    sleep 1

done

wait
