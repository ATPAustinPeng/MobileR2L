from data import (
    PositionalEmbedder
)
from model.R2L import R2L

import click
from easydict import EasyDict as edict

@click.command()
@click.option('--project_name', type=str)
# dataset 
@click.option('--root_dir', type=str)
@click.option('--dataset_type', type=click.Choice(['Blender', 'Colmap'], case_sensitive=True))
@click.option('--pseudo_dir', type=str)
@click.option('--ff', is_flag=True, default=False, help='Whether the scene is forward-facing')
@click.option('--ndc', is_flag=True, default=False)
@click.option('--scene', type=str)
@click.option('--testskip', type=int, default=8)
@click.option('--factor', type=int, default=4)
@click.option('--llffhold', type=int, default=8)
@click.option('--bd_factor', type=float, default=0.75)
@click.option('--camera_convention', type=str, default='openGL')
# train/test 
@click.option('--run_train', is_flag=True)
@click.option('--run_render', is_flag=True)
@click.option('--render_test', is_flag=True, help='Render the testset.')
@click.option('--finetune', is_flag=True)
@click.option('--amp', is_flag=True, default=False)
@click.option('--resume', is_flag=True, default=False)
@click.option('--perturb', is_flag=True, default=True)
@click.option('--num_workers', type=int)
@click.option('--batch_size', type=int)
@click.option('--num_epochs', type=int, default=500000)
@click.option('--num_iters', type=int, default=500000)
@click.option('--ckpt_dir', type=str)
@click.option('--lrate', type=float, default=0.0005)
@click.option('--lr_scale', type=float, default=1.0)
@click.option('--lrate_decay', type=int, default=500)
@click.option('--warmup_lr', type=str, default='0.0001,200')
@click.option('--lpips_net', type=str, default='alex')
@click.option('--export_onnx', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
@click.option('--no_cache', is_flag=True, default=False)
@click.option('--convert_snap_gelu', type=bool, default=True)
# model
@click.option('--input_height', type=int)
@click.option('--input_width', type=int)
@click.option('--output_height', type=int)
@click.option('--output_width', type=int)
@click.option('--n_sample_per_ray', type=int, default=8)
@click.option('--multires', type=int, default=6)
@click.option('--num_sr_blocks', type=int, default=3)
@click.option('--num_conv_layers', type=int, default=2)
@click.option('--sr_kernel', type=(int, int, int), default=(64, 64, 16))
@click.option('--netdepth', type=int, default=60)
@click.option('--netwidth', type=int, default=256)
@click.option('--activation_fn', type=str, default='gelu')
@click.option('--use_sr_module', is_flag=True, default=True)
# logging/saving options
@click.option("--i_print", type=int, default=10000, help='frequency of console printout and metric loggin')
@click.option("--train_image_log_step", type=int, default=5000, help='frequency of tensorboard image logging')
@click.option("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
@click.option("--i_save_rendering", type=int, default=10000, help='frequency of weight ckpt saving')
@click.option("--i_testset", type=int, default=10000, help='frequency of testset saving')
@click.option("--i_video", type=int, default=100000, help='frequency of render_poses video saving')
def main(**kwargs):
    # setup args
    args = edict(kwargs)
    print(args)

    from model.preresnetcs import preresnet20cs
    model = preresnet20cs(depth=20, num_classes=10)
    for m in model.modules():
        print(model)


    # embedder = PositionalEmbedder(args.multires, 'cpu', True)
    # model = R2L(args, 3 * args.n_sample_per_ray * embedder.embed_dim, 3)

    # for m in model.modules():
    #     print(m)
    
    # print('\n\n\n\n\n\n\n\n\n\n\n')

    # for k, v in model.named_modules():
    #     print(k)
    #     print("########################################")
    #     print(v)
    #     print("\n")

if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    # with logging_redirect_tqdm():
    main()