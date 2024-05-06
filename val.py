from models.frameworks import get_model
from models.base import get_optimizer, get_scheduler
from utils import rend_util, train_util, mesh_util, io_util
from utils.dist_util import get_local_rank, init_env, is_master, get_rank, get_world_size
from utils.print_fn import log
from utils.logger import Logger
from utils.checkpoints import CheckpointIO
from dataio import get_data

import os
import sys
import time
import functools
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import autograd

import numpy as np
import json

def validate_all_normals(self):
        total_MAE = 0
        idxs = [i for i in range(self.dataset.n_images)]
        f = open(os.path.join(self.base_exp_dir, 'result_normal.txt'), 'a')
        for idx in idxs:
            normal_maps, color_fine = self.validate_image(idx, resolution_level=1, only_normals=True)
            try:
                GT_normal = torch.from_numpy(self.dataset.normal_np[idx])
                cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
                cos_loss = cos(normal_maps.view(-1, 3), GT_normal.view(-1, 3))
                cos_loss = torch.clamp(cos_loss, (-1.0 + 1e-10), (1.0 - 1e-10))
                loss_rad = torch.acos(cos_loss)
                loss_deg = loss_rad * (180.0 / np.pi)
                total_MAE += loss_deg.mean()
                f.write(str(idx) + '_MAE:')
                f.write(str(loss_deg.mean().data.item()) + '    ')
                f.write('\n')
                f.flush()
            except:
                continue
        MAE = total_MAE / self.dataset.n_images
        f.write('\n')
        f.write('MAE_final:')
        f.write(str(MAE.data.item()) + '    ')
        f.close()

def main_function(args):
    args.device_ids = list(range(torch.cuda.device_count()))
    init_env(args)
    #----------------------------
    #-------- shortcuts ---------
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    exp_dir = args.training.exp_dir
    mesh_dir = os.path.join(exp_dir, 'meshes')
    
    device = torch.device('cuda', local_rank)

    logger = Logger(
        log_dir=exp_dir,
        img_dir=os.path.join(exp_dir, 'imgs'),
        monitoring=args.training.get('monitoring', 'tensorboard'),
        monitoring_dir=os.path.join(exp_dir, 'events'),
        rank=rank, is_master=is_master(), multi_process_logging=(world_size > 1))

    log.info("=> Experiments dir: {}".format(exp_dir))
   
    val_dataset = get_data(args, downscale=1.0)
    bs = args.data.get('batch_size', None)
    if args.ddp:
        val_sampler = DistributedSampler(val_dataset)
        valloader = torch.utils.data.DataLoader(val_dataset, sampler=val_sampler, batch_size=bs)
    else:
        valloader = DataLoader(val_dataset,
            batch_size=1,
            shuffle=True)
    
    # Create model
    model, trainer, render_kwargs_train, render_kwargs_test, volume_render_fn = get_model(args)
    model.to(device)
    log.info(model)
    log.info("=> Nerf params: " + str(train_util.count_trainable_parameters(model)))

    render_kwargs_test['H'] = val_dataset.H
    render_kwargs_test['W'] = val_dataset.W

    # build optimizer
    optimizer = get_optimizer(args, model)

    # checkpoints
    checkpoint_io = CheckpointIO(checkpoint_dir=os.path.join(exp_dir, 'ckpts'), allow_mkdir=is_master())
    if world_size > 1:
        dist.barrier()
    # Register modules to checkpoint
    checkpoint_io.register_modules(
        model=model,
        optimizer=optimizer,
    )

    # Load checkpoints
    load_dict = checkpoint_io.load_file(
        args.training.ckpt_file,
        ignore_keys=args.training.ckpt_ignore_keys,
        only_use_keys=args.training.ckpt_only_use_keys,
        map_location=device)


    it = load_dict.get('global_step', 0)

    # pretrain if needed. must be after load state_dict, since needs 'is_pretrained' variable to be loaded.
    #---------------------------------------------
    #-------- init perparation only done in master
    #---------------------------------------------

    # Parallel training
    if args.ddp:
        trainer = DDP(trainer, device_ids=args.device_ids, output_device=local_rank, find_unused_parameters=False)
    log.info('=> Start Validating..., it={}, in {}'.format(it, exp_dir))

    total_MAE = 0
    os.makedirs(os.path.join(exp_dir,f'imgs/eval'), exist_ok=True)
    f=open(os.path.join(exp_dir,f'imgs/eval/MAE.txt'), 'w+')
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    for i in tqdm(range(len(valloader))):
    # for i in tqdm(range(1)):
        #-------------------
        # validate
        #-------------------
        with torch.no_grad():
            (val_ind, val_in, val_gt) = next(iter(valloader))
            val_ind = val_ind.item()
            intrinsics = val_in["intrinsics"].to(device)
            c2w = val_in['c2w'].to(device)
            
            # N_rays=-1 for rendering full image
            rays_o, rays_d, select_inds = rend_util.get_rays(
                c2w, intrinsics, render_kwargs_test['H'], render_kwargs_test['W'], N_rays=-1, opengl=args.data.opengl)
            target_rgb = val_gt['rgb'].to(device)
            render_kwargs_test['cone_angle'] = intrinsics[0,0,0].item()
            rgb, depth_v, ret = volume_render_fn(rays_o, rays_d, c2w=c2w.expand(*rays_d.shape[:-1], 4, 4), calc_normal=True, detailed_output=True, **render_kwargs_test)

            to_img = functools.partial(
                rend_util.lin2img, 
                H=render_kwargs_test['H'], W=render_kwargs_test['W'],
                batched=render_kwargs_test['batched'])

            # logger.add_imgs(to_img(target_rgb), 'val/gt_rgb', val_ind)
            # logger.add_imgs(to_img(rgb), 'val/predicted_rgb', val_ind)
            # logger.add_imgs(to_img((rgb-target_rgb).abs()), 'val/rgb_error_map', val_ind)
            # logger.add_imgs(to_img((depth_v/(depth_v.max()+1e-10)).unsqueeze(-1)), 'val/pred_depth_volume', val_ind)
            # logger.add_imgs(to_img(ret['mask_volume'].unsqueeze(-1)), 'val/pred_mask_volume', it)
            # if 'depth_surface' in ret:
            #     logger.add_imgs(to_img((ret['depth_surface']/ret['depth_surface'].max()).unsqueeze(-1)), 'val/pred_depth_surface', val_ind)
            # if 'mask_surface' in ret:
            #     logger.add_imgs(to_img(ret['mask_surface'].unsqueeze(-1).float()), 'val/predicted_mask', val_ind)
            # if hasattr(trainer, 'val'):
            #     trainer.val(logger, ret, to_img, it, render_kwargs_test)
            
            # ADD: Polarization Validation
            # if hasattr(trainer, 'val_pol') and render_kwargs_test['has_pol']:
            #     from models.frameworks.pnr import indexing_2d_samples
            #     AoP_map = val_gt['AoP_map'].to(device)
            #     DoP_map = val_gt['DoP_map'].to(device)
            #     aop_sample_idx = indexing_2d_samples(select_inds, render_kwargs_test['H'], render_kwargs_test['W'], 
            #                                          args.model.get('gaussian_scale_factor', 1.0)).reshape(1,-1)
            #     aop_samples = torch.gather(val_gt['AoP_map'].to(device), 1 , aop_sample_idx.long()).reshape(*AoP_map.shape, 4)
            #     mask = val_gt['mask'].to(device)
            #     gt = {}
            #     gt['AoP_map']= AoP_map
            #     gt['DoP_map']= DoP_map
            #     gt['rgb'] = target_rgb
            #     gt['mask'] = mask
            #     gt['aop_samples'] = aop_samples
            #     trainer.val_pol(logger, ret, c2w, gt, to_img, val_ind, render_kwargs_test)
            
            # Validate Normals
            pred_normals = ret['normals_volume']
            gt_normals = val_dataset.normals[val_ind].reshape(1,-1,3)

            # BGR to RGB (OpenCV Legacy)
            gt_normals = gt_normals[...,[2,1,0]]
            # Flip XZ (Mitsuba)
            gt_normals[...,[0,2]] *= -1

            num_pixel = gt_normals.shape[1]
            validate_mask = val_gt['mask']
            pred_normals_, gt_normals_ = pred_normals[validate_mask,:].cpu(), gt_normals[validate_mask,:]
            cos_loss = cos(pred_normals_.view(-1, 3), gt_normals_.view(-1, 3))
            cos_loss = torch.clamp(cos_loss, (-1.0 + 1e-10), (1.0 - 1e-10))
            loss_rad = torch.acos(cos_loss)
            loss_deg = (loss_rad * (180.0 / np.pi)).sum() / num_pixel
            total_MAE += loss_deg
            pred_normals_img = (ret['normals_volume']/2.+0.5)
            pred_normals_img[~validate_mask,:] = 0. 
            logger.add_imgs(to_img(pred_normals_img), 'eval/predicted_normals', val_ind)
            logger.add_imgs(to_img(gt_normals/2.+0.5), 'eval/gt_normals', val_ind)
            f.write(str(val_ind) + '_MAE:')
            f.write(str(loss_deg.data.item()) + '    ')
            f.write('\n')
            f.flush()

    MAE = total_MAE / len(valloader)   
    f.write('\n')
    f.write('MAE_final:')
    f.write(str(MAE.data.item()) + '    ')
    f.close()

    #-------------------
    # validate mesh
    #-------------------
    # if is_master():

    #     with torch.no_grad():
    #         io_util.cond_mkdir(mesh_dir)
    #         mesh_util.extract_mesh(
    #             model.implicit_surface, 
    #             filepath=os.path.join(mesh_dir, '{:08d}.ply'.format(it)),
    #             volume_size=args.data.get('volume_size', 2.0),
    #             show_progress=is_master())


    log.info("Everything done.")

if __name__ == "__main__":
    # Arguments
    parser = io_util.create_args_parser()
    parser.add_argument("--ddp", action='store_true', help='whether to use DDP to train.')
    parser.add_argument("--port", type=int, default=None, help='master port for multi processing. (if used)')
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown, base_config_path= None if args.base_config is None else args.base_config)
    main_function(config)