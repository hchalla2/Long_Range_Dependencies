#!/bin/bash

#SBATCH -N 1
#SBATCH -t 04:00:00                      # wall time (D-HH:MM:SS)
#SBATCH -p htc
#SBATCH -G a100:4
#SBATCH --mem 0 
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment
module load mamba/latest
source activate myenv

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_SIZE=7B
NUM_GPUS=4
BATCH_SIZE_PER_GPU=2    
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file stage3_offloading_accelerate.conf \
    finetune.py \
    --model_name_or_path EleutherAI/gpt-neo-1.3B \
    --tokenizer_name EleutherAI/gpt-neo-1.3B \
    --train_file Cohort_train.csv \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 4 \
    --output_dir ouptut_Cohort_dir/ \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1
    
