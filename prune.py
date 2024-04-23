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

    # load pretrained model (passed through args)
    engine = R2LEngine(dataset_info, logger, args)
    model = engine.engine

    # # get/save preprune stats
    # engine.save_arch("original_arch.txt")
    # engine.save_prune_txt(mode="PREPRUNE")
    if args.get_info:
        from ptflops import get_model_complexity_info
        input_size = (312, 100, 100)
        macs, params = get_model_complexity_info(model, input_size)#, as_strings=True)#, print_per_layer_stat=True)

        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2

        print(f"Model Size: {size_all_mb} MB")
        print(f"Number of MACs: {macs}")
        print(f"Number of parameters: {params}")

        exit()

    if args.export_onnx:
        engine.export_onnx()
        exit(0)

    ############################## PREPRUNE ##############################
    # perform test and video rendering; export onnx for coremltools
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

        # engine.export_onnx(extra_path="-preprune")
        
    # https://github.com/Eric-mingjie/rethinking-network-pruning/blob/master/cifar/network-slimming/resprune.py
    logger.info("Calculate pruning config.\n")

    # count # of BN weights
    total = 0
    for n, m in model.named_modules():
        if 'tail' in n:
            continue

        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            total += m.weight.data.shape[0]

    # collect BN weights/scale factor
    bn = torch.zeros(total)
    index = 0
    for n, m in model.named_modules():
        if 'tail' in n:
            continue
            
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size

    # sort BN scale factors
    y, i = torch.sort(bn)
    thre_index = int(total * args.prune_percentage / 100)
    thre = y[thre_index]

    # prune channels based on BN scale factors
    pruned = 0
    cfg = []
    cfg_mask = []
    for k, (n, m) in enumerate(model.named_modules()):
        if 'tail' in n:
            continue
            
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().to(device)
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            m.weight.data.mul_(mask)
            m.bias.data.mul_(mask)
            cfg.append(int(torch.sum(mask)))
            cfg_mask.append(mask.clone())
            logger.info(f"layer index: {k} \t total channel: {mask.shape[0]} \t remaining channel: {int(torch.sum(mask))}")
        elif isinstance(m, nn.MaxPool2d):
            cfg.append('M')

    pruned_ratio = pruned/total
    logger.info(f"Pruned ratio: {pruned_ratio}")
    logger.info(f"CFG: {cfg}")
    
    # ensure all devices have the CFG before initializing new model
    dist.barrier()

    # initialize new model
    newengine = R2LEngine(dataset_info, logger, args, cfg=cfg)
    newmodel = newengine.engine

    old_modules = list()
    new_modules = list()
    for n, m in model.named_modules():
        if 'tail' in n:
            continue
        old_modules.append(m)

    for n, m in newmodel.named_modules():
        if 'tail' in n:
            continue
        new_modules.append(m)

    layer_id_in_cfg = 0
    start_mask = torch.ones(3)
    end_mask = cfg_mask[layer_id_in_cfg]
    conv_count = 0

    for layer_id in range(len(old_modules)):
        m0 = old_modules[layer_id]
        m1 = new_modules[layer_id]
        if isinstance(m0, nn.BatchNorm2d) or isinstance(m0, nn.SyncBatchNorm):
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if isinstance(old_modules[layer_id + 1], channel_selection):
                # If the next layer is the channel selection layer, then the current batchnorm 2d layer won't be pruned.
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                # We need to set the channel selection layer.
                m2 = new_modules[layer_id + 1]
                m2.indexes.data.zero_()
                m2.indexes.data[idx1.tolist()] = 1.0

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
            else:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Conv2d):
            if conv_count == 0:
                m1.weight.data = m0.weight.data.clone()
                conv_count += 1
                continue
            if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d) or isinstance(old_modules[layer_id-1], nn.SyncBatchNorm):
                # This convers the convolutions in the residual block.
                # The convolutions are either after the channel selection layer or after the batch normalization layer.
                conv_count += 1
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()

                # IMPORTANT: If the current convolution is not the last convolution in the residual block, then we can change the 
                # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
                if conv_count % 2 != 1:
                    w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()
                continue

            # We need to consider the case where there are downsampling convolutions. 
            # For these convolutions, we just copy the weights.
            m1.weight.data = m0.weight.data.clone()
        elif isinstance(m0, nn.Linear):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))

            m1.weight.data = m0.weight.data[:, idx0].clone()
            m1.bias.data = m0.bias.data.clone()

    logger.info("Successfully pruned model.")

    del engine
    del model

    engine = newengine

    # # model should be different now
    # engine.save_arch("pruned_arch.txt")
    # engine.save_prune_txt(mode="POSTPRUNE", cfg=cfg)
    if args.get_pruned_info:
        from ptflops import get_model_complexity_info
        input_size = (312, 100, 100)
        macs, params = get_model_complexity_info(newmodel, input_size)#, as_strings=True)#, print_per_layer_stat=True)

        param_size = 0
        for param in newmodel.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in newmodel.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2

        print(f"Model Size: {size_all_mb} MB")
        print(f"Number of MACs: {macs}")
        print(f"Number of parameters: {params}")

        exit()

    # save cfg
    # if is_main_process():
        # with open(os.path.join(exp_weight_dir, 'cfg.json'), "w") as file:
        #     json.dump(cfg, file)

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
        # dist.barrier()
        # engine.export_onnx(extra_path="-postprune")

    # TODO save pruned model
    # train pruned model
    if args.run_train:
        pseudo_dataloader, num_pseudo = get_pseduo_dataloader(
            args.pseudo_dir,
            args.batch_size,
            args.num_workers,
            args.camera_convention,
            dataset_info.sc
        )
        global_step = engine.buffer.start
        with tqdm(
            range(
                set_epoch_num(
                    global_step,
                    global_step + args.num_iters,# args.num_iters, #modify to train for a set number of iterations
                    args.batch_size,
                    num_pseudo,
                    world_size
                )
            ),
            ascii=True,
            ncols=120,
            disable=not is_main_process()
        ) as pbar:
            for i in pbar:
                print(f"EPOCH {i} | GLOBAL_STEP {global_step}")
                pseudo_dataloader.sampler.set_epoch(i)
                for _, (rays_o, rays_d, target_rgb) in enumerate(pseudo_dataloader):
                    global_step += 1
                    loss, psnr, best_psnr = engine.train_step(
                        rays_o=rays_o.to(device),
                        rays_d=rays_d.to(device),
                        target_rgb=target_rgb.to(device),
                        global_step=global_step
                    )
                    if global_step % args.i_video == 0 and video_poses is not None:
                        engine.render(
                            c2ws=video_poses,
                            gt_imgs=None,
                            global_step=global_step,
                            save_video=True
                        )
                    
                    if global_step % args.i_testset == 0:
                        engine.render(
                            c2ws=test_poses,
                            gt_imgs=test_images,
                            global_step=global_step,
                            save_rendering=(global_step % args.i_save_rendering == 0)
                        )  
                    pbar.set_postfix(iter=global_step, loss=loss.item(), psnr=psnr, best_psnr=best_psnr) 
                    dist.barrier()

        engine.export_onnx()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        main()