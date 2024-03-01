
python3 main.py \
    --dataset_type Blender \
    --root_dir dataset/nerf_synthetic \
    --run_train \
    --num_workers 12 \
    --batch_size 10 \
    --num_iters 600000 \
    --input_height 100 \
    --input_width 100 \
    --output_height 800 \
    --output_width 800 \
    --i_testset 10000 \
    --amp \
    --lrate 0.0005 
