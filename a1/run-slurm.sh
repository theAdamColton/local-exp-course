#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH --mail-user=u1377031@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_1-%j


#source ~/miniconda3/etc/profile.d/conda.sh
#conda activate base

#conda info

#CACHE_DIR=/scratch/general/vast/u1377031/huggingface_cache
#mkdir -p ${CACHE_DIR}
#export TRANSFORMER_CACHE=${CACHE_DIR}
OUT_DIR=./out/
mkdir -p ${OUT_DIR}

python main.py \
	--output_dir ${OUT_DIR} \
	--bf16 \
	--max_length 768 \
	--per_device_train_batch_size 16 \
	--per_device_eval_batch_size 16 \
	--evaluation_strategy epoch \
	--eval_steps 1 \
	--do_eval true \
	--save_total_limit 1 \
	--overwrite_output_dir \
	--logging_steps 1 \
	--gradient_accumulation_steps 1 \
	--optim paged_adamw_32bit \
	--learning_rate 2e-5 \
	--seed 42 \
	--gradient_checkpointing true \
	--log_level info \
	--warmup_steps 100 \
	--lr_scheduler_type cosine \
	--num_train_epochs 1.0 \
	--max_grad_norm 1.0 \
	--report_to wandb

