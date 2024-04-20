#!/bin/bash
#SBATCH --job-name=log_r2l_model.sh
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH -t 0-04:00:00
#SBATCH --gres=gpu:1

# sbatch pace-script/log_r2l_model.sh 1 lego

module load anaconda3
conda deactivate
conda activate r2l

cd /home/hice1/apeng39/scratch/SENeLF/MobileR2L

nGPU=$1 #1
scene=$2 #"lego"
project_name="${scene}-pruned"

python3 log_r2l_model.py \
    --dataset_type Blender \
    --pseudo_dir model/teacher/ngp_pl/Pseudo/$scene  \
    --root_dir dataset/nerf_synthetic \
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
    --i_video 10000 \
    --amp \
    --lrate 0.0005 \
    --ckpt_dir $ckpt_dir \
    --resume