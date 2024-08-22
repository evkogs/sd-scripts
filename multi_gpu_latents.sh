#!/bin/bash
#SBATCH --job-name=cache_latents
#SBATCH --output=O-%x.%j
#SBATCH --error=E-%x.%j
#SBATCH --partition=slurm_rtx4090
#SBATCH --cpus-per-gpu=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --qos=gpu_qos

TRAIN_DATA_DIR=/mnt/s3fs_cache/training_data


TOTAL_SPLIT=64 # Total number of processes
OFFSET=0
CUDA_DEVICES_NUM=8 # Number of GPUs
PYTHON=python
MODEL_PATH=stabilityai/stable-diffusion-xl-base-1.0

# Launch 32 processes, 3 per GPU
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
        --bucket_no_upscale \
        --full_path \
        --skip_existing \
        --recursive \
        --train_data_dir $TRAIN_DATA_DIR &

done

wait
