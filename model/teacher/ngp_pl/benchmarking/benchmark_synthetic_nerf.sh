#!/bin/bash
#SBATCH --job-name=benchmarking_synthetic_nerf.sh
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32GB

conda deactivate
conda activate ngp_pl_cu117_sm80

cd /home/hice1/apeng39/scratch/SENeLF/MobileR2L/model/teacher/ngp_pl

export ROOT_DIR=../../../dataset/nerf_synthetic/
scene=$1

python3 train.py \
    --root_dir $ROOT_DIR/$scene \
    --exp_name $scene  \
    --num_epochs 30 --batch_size 16384 --lr 2e-2 --eval_lpips \
    --num_gpus 1

