#!/bin/bash

#SBATCH -N 1
#SBATCH -t 04:00:00                      # wall time (D-HH:MM:SS)
#SBATCH -p htc
#SBATCH -G a100:1
#SBATCH --mem 0 
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment
#SBATCH -C a100_80
module load mamba/latest
source activate myenv

export CUDA_VISIBLE_DEVICES=0

MODEL_SIZE=7B
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training model using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


python hf_inference.py --name_or_path ouptut_dir --temperature 0.7 \
    --top_p 0.95     --top_k 50     --seed 1     --max_batch_size 2 \
    --model_dtype bf16     --attn_impl torch     --max_seq_len 2048  \
    --use_cache True     --max_new_tokens 2048     --eos_token_id 2 \
    --repetition_penalty 1     --input_data_path smoking_test.csv   --output_data_path pred_test.csv


