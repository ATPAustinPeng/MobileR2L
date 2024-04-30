#!/bin/bash
#SBATCH --job-name=export_pruned_model.sh
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
project_name="${scene}-pruned"
ckpt_dir=$3

ncpu_cores=$(nproc --all)
omp_num_threads=$((ncpu_cores / nGPU))

# BOTH run_render and run_train flags need to be True
# run_train puts model in DDP (which must happen as weights were saved as DDP)

# occassionally add --master_port=25641 if you are getting socket issues
# [W socket.cpp:464] [c10d] The server socket has failed to bind to [::]:25641 (errno: 98 - Address already in use).
# [W socket.cpp:464] [c10d] The server socket has failed to bind to 0.0.0.0:25641 (errno: 98 - Address already in use).
# [E socket.cpp:500] [c10d] The server socket has failed to listen on any local network address.

OMP_NUM_THREADS=$omp_num_threads torchrun --nproc_per_node=$nGPU --master_port=25641 export_pruned_model.py \
    --project_name $project_name \
    --dataset_type Blender \
    --pseudo_dir model/teacher/ngp_pl/Pseudo/$scene  \
    --root_dir dataset/nerf_synthetic \
    --run_render \
    --num_workers 12 \
    --batch_size 10 \
    --num_iters 100000 \
    --input_height 100 \
    --input_width 100 \
    --output_height 800 \
    --output_width 800 \
    --scene $scene \
    --i_weights 1000 \
    --i_testset 10000 \
    --i_save_rendering 10000 \
    --i_video 10000 \
    --amp \
    --lrate 0.0005 \
    --ckpt_dir $ckpt_dir \
    --resume