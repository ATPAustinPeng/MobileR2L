try:
    import pretty_traceback
    pretty_traceback.install()
except ImportError:
    pass

import os
import click
import torch
import logging
from tqdm import tqdm
from pprint import pprint
import torch.distributed as dist
from easydict import EasyDict as edict
from tqdm.contrib.logging import logging_redirect_tqdm

from data import (
    select_and_load_dataset,
    get_pseduo_dataloader,
    PositionalEmbedder
)
from utils import (
    is_main_process,
    set_epoch_num,
    get_rank,
    init_distributed_mode,
    get_world_size,
    main_process
)
from model import R2LEngine
from model.R2L import R2L
from model.R2LCFG import R2LCFG
from model.channel_selection import channel_selection

# for pruning
import torch.nn as nn
import numpy as np
import json

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
# prune
@click.option('--prune_percentage', type=int, default=30)
@click.option('--get_pruned_info', is_flag=True, default=False, help="Get pruned model info")
@click.option('--get_info', is_flag=True, default=False, help="Get original model info.")
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
@click.option("--i_save_rendering", type=int, default=10000, help='frequency of gt vs model rendered img saving')
@click.option("--i_testset", type=int, default=10000, help='frequency of testset saving')
@click.option("--i_video", type=int, default=100000, help='frequency of render_poses video saving')
def main(**kwargs):
    torch.backends.cudnn.benchmark = True
    # setup args
    args = edict(kwargs)
    print(args)

    init_distributed_mode(args)
    device = get_rank()
    world_size = get_world_size()
    
    # load data    
    dataset_info = select_and_load_dataset(
        basedir=os.path.join(args.root_dir, args.scene),
        dataset_type=args.dataset_type,
        input_height=args.input_height,
        input_width=args.input_width,
        output_height=args.output_height,
        output_width=args.output_width,
        scene=args.scene,
        test_skip=args.testskip,
        factor=args.factor,
        bd_factor=args.bd_factor,
        llffhold=args.llffhold,
        ff=args.ff,
        use_sr_module=args.use_sr_module,
        camera_convention=args.camera_convention,
        ndc=args.ndc,
        device=device,
        n_sample_per_ray=args.n_sample_per_ray
    )
    i_test = dataset_info.i_split.i_test
    test_images = dataset_info.images[i_test]
    test_poses = dataset_info.poses[i_test]
    video_poses = dataset_info.render_poses

    logger = logging.getLogger(__name__)
    pprint(args)

    # get pruned cfg
    cfg_dir = os.path.dirname(args.ckpt_dir)
    cfg_path = os.path.join(cfg_dir, 'cfg.json')

    with open(cfg_path, 'r') as f:
        data = f.read()

    cfg = eval(data)
    logger.info(cfg)

    # get cfg mask
    cfg_mask_path = os.path.join(cfg_dir, 'cfg_mask.pth')
    cfg_mask = torch.load(cfg_mask_path)

    # load pruned model (passed through args)
    engine = R2LEngine(dataset_info, logger, args, cfg=cfg)

    dist.barrier()

    modules = list()
    for n, m in engine.engine.named_modules():
        if 'tail' in n:
            continue
        modules.append(m)

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0

    for layer_id in range(len(modules)):
        m0 = modules[layer_id]
        # m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d) or isinstance(m0, nn.SyncBatchNorm):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            # set the channel selection layer, if it is next, using cfg_mask
            if isinstance(modules[layer_id + 1], channel_selection):
                m1 = modules[layer_id + 1]
                m1.indexes.data.zero_()
                m1.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]

    logger.info("Successfully pruned model.")

    print(engine.engine)

    # perform testing postprune and save onnx
    if args.run_render:
        logger.info('Starting rendering. \n')
        # render testset
        engine.render(
            c2ws=test_poses,
            gt_imgs=test_images,
            global_step=0,
            save_rendering=True
        )
        # render videos
        if video_poses is not None:
            engine.render(
                c2ws=video_poses,
                gt_imgs=None,
                global_step=0,
                save_video=True
            )

        engine.export_onnx(extra_path="-postprune")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        main()