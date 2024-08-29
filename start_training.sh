#!/bin/bash
#SBATCH --job-name=sdxl_train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Load the conda environment
module load conda
conda activate diffusion_torch2.5

# Set environment variables
export FI_EFA_USE_DEVICE_RDMA=1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export NCCL_TUNER_PLUGIN=/opt/aws-ofi-nccl/lib/libnccl-ofi-tuner.so
export NCCL_NET_GDR_LEVEL=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_LOG=perf_hints
export TORCH_DISTRIBUTED_DEBUG=OFF

# Run the job
srun /opt/amazon/openmpi/bin/mpirun \
    --hostfile /home/ubuntu/efs_gpu/libs/my-hosts \
    -n 16 -N 4 \
    -x FI_EFA_USE_DEVICE_RDMA=1 \
    -x LD_LIBRARY_PATH \
    -x NCCL_DEBUG \
    -x NCCL_TUNER_PLUGIN \
    -x NCCL_NET_GDR_LEVEL \
    -x PYTORCH_CUDA_ALLOC_CONF \
    -x TORCH_LOG \
    -x TORCH_DISTRIBUTED_DEBUG \
    --mca pml ^cm --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to numa \
    accelerate launch \
    /home/ubuntu/efs_gpu/libs/sd-scripts/sdxl_train.py \
    --enable_bucket \
    --min_bucket_reso=640 \
    --max_bucket_reso=1536 \
    --pretrained_model_name_or_path=/home/ubuntu/efs_gpu/checkpoints/oils_v2_full/oils_v2_full_25-step00070000.safetensors \
    --train_data_dir=/mnt/s3fs_cache/training_data_cool \
    --resolution=1024,1024 \
    --output_dir=/home/ubuntu/efs_gpu/checkpoints/oils_v2_full \
    --logging_dir=/mnt/s3fs_cache/training_data_cool/logs \
    --save_model_as=safetensors \
    --train_batch_size=8 \
    --learning_rate=1e-7 \
    --lr_warmup_steps=2000 \
    --gradient_accumulation_steps=1 \
    --lr_scheduler_num_cycles=1 \
    --no_half_vae \
    --max_train_steps=50000 \
    --lr_scheduler linear \
    --mixed_precision=fp16 \
    --weight_dtype=fp32 \
    --fp32_weights \
    --save_precision=fp16 \
    --caption_extension=.txt \
    --cache_latents \
    --cache_latents_to_disk \
    --optimizer_type=AdamW \
    --optimizer_args weight_decay=0.01 betas="(0.9, 0.995)" \
    --max_grad_norm=0.3 \
    --max_data_loader_n_workers=0 \
    --bucket_reso_steps=64 \
    --save_every_n_steps=500 \
    --sdpa \
    --min_snr_gamma=9 \
    --enable_wildcard \
    --loss_type=l2 \
    --noise_offset=0.05 \
    --caption_dropout_rate=0.15 \
    --log_with wandb \
    --wandb_api_key=797aed11a5c9a0e8bf5fbe706e32ccfdb20af817 \
    --output_name=oils_v2_full_29_cool \
    --max_token_length=225 \
    --skip_file_existence_check \
    --sample_at_first \
    --sample_every_n_steps=500 \
    --sample_sampler=dpmsolver++ \
    --sample_prompts=/home/ubuntu/s3_model_checkpoints/oils/sample_prompts.txt \
    --deepspeed \
    --zero_stage 1
