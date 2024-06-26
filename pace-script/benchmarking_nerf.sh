#!/bin/bash
#SBATCH --job-name=benchmarking_nerf.sh
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

nGPU=$1
scene=$2
ncpu_cores=$(nproc --all)
omp_num_threads=$((ncpu_cores / nGPU))

OMP_NUM_THREADS=$omp_num_threads python3 -m torch.distributed.launch --nproc_per_node=$nGPU --master_port=25641 --use_env main.py \
    --project_name $scene \
    --dataset_type Blender \
    --pseudo_dir model/teacher/ngp_pl/Pseudo/$scene  \
    --root_dir dataset/nerf_synthetic \
    --run_train \
    --num_workers 12 \
    --batch_size 10 \
    --num_iters 600000 \
    --input_height 100 \
    --input_width 100 \
    --output_height 800 \
    --output_width 800 \
    --scene $scene \
    --i_weights 1000 \
    --i_testset 10000 \
    --i_video 50000 \
    --amp \
    --lrate 0.0005
