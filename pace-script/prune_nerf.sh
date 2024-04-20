#!/bin/bash
#SBATCH --job-name=benchmarking_pruned_nerf.sh
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:H100:4
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=12

module load anaconda3
conda deactivate
conda activate r2l

cd /home/hice1/apeng39/scratch/SENeLF/MobileR2L

nGPU=$1 #1
scene=$2 #"lego"
project_name="${scene}-pruned"
ckpt_dir="/home/hice1/apeng39/scratch/SENeLF/MobileR2L/test_prune/test_prune_ckpt.tar"
ncpu_cores=$(nproc --all)
omp_num_threads=$((ncpu_cores / nGPU))

# BOTH run_render and run_train flags need to be True
# run_train puts model in DDP (which must happen as weights were saved as DDP)
# python3 prune.py \

# Note: for prune.py ONLY, num_iters = number of additional iters to train on top of checkpoint

OMP_NUM_THREADS=$omp_num_threads python3 -m torch.distributed.launch --nproc_per_node=$nGPU --use_env prune.py \
    --dataset_type Blender \
    --pseudo_dir model/teacher/ngp_pl/Pseudo/$scene  \
    --root_dir dataset/nerf_synthetic \
    --run_train \
    --run_render \
    --num_workers 12 \
    --batch_size 10 \
    --num_iters 80000 \
    --input_height 100 \
    --input_width 100 \
    --output_height 800 \
    --output_width 800 \
    --scene $scene \
    --i_weights 1000 \
    --i_testset 10000 \
    --i_video 10000 \
    --amp \
    --lrate 0.0005 \
    --prune_percentage 30 \
    --ckpt_dir $ckpt_dir \
    --resume


# nGPU=$1
# scene=$2
# ckpt_dir=$3
# ncpu_cores=$(nproc --all)
# omp_num_threads=$((ncpu_cores / nGPU))

# OMP_NUM_THREADS=$omp_num_threads python3 -m torch.distributed.launch --nproc_per_node=$nGPU --use_env main.py \
#     --project_name $scene \
#     --dataset_type Blender \
#     --pseudo_dir model/teacher/ngp_pl/Pseudo/$scene  \
#     --root_dir dataset/nerf_synthetic \
#     --run_train \
#     --num_workers 12 \
#     --batch_size 10 \
#     --num_iters 600000 \
#     --input_height 100 \
#     --input_width 100 \
#     --output_height 800 \
#     --output_width 800 \
#     --scene $scene \
#     --i_weights 1000 \
#     --i_testset 10000 \
#     --amp \
#     --lrate 0.0005
