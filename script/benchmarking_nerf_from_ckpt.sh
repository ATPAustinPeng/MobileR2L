nGPU=$1
scene=$2
ckpt_dir=$3
ncpu_cores=$(nproc --all)
omp_num_threads=$((ncpu_cores / nGPU))

OMP_NUM_THREADS=$omp_num_threads python3 -m torch.distributed.launch --nproc_per_node=$nGPU --use_env main.py \
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
    --amp \
    --lrate 0.0005 \
    --ckpt_dir $ckpt_dir \
    --resume
