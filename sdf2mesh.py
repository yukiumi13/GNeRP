import torch, os
import trimesh
import mcubes
import argparse
import json
import numpy as np

from utils.print_fn import log
import utils.general as utils
import utils.plots as plt
from utils import rend_util
from utils.io_util import load_yaml
from models.frameworks import get_model

def scale_anything(dat, inp_scale, tgt_scale):
    if inp_scale is None:
        inp_scale = [dat.min(), dat.max()]
    dat = (dat  - inp_scale[0]) / (inp_scale[1] - inp_scale[0])
    dat = dat * (tgt_scale[1] - tgt_scale[0]) + tgt_scale[0]
    return dat

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)
    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry_(bound_min, bound_max, resolution, threshold, query_func):
    log.info('Threshold: {}'.format(threshold))
    sdfs = extract_fields(bound_min, bound_max, resolution, query_func)
    log.info('Marching Cubes')
    vertices, triangles = mcubes.marching_cubes(sdfs, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles

def extract_geometry(model, bound_min, bound_max, resolution, threshold=0.0):
    """Coarse-to-Fine Mesh Extraction

    Args:
        model: neural sdf model
        bound_min: aabb
        bound_max: aabb
        resolution: grid vertices
        threshold: level set. Defaults to 0.0.

    Returns:
        mesh
    """        
    
    return extract_geometry_(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -model.forward_surface(pts))

def validate_mesh(model, epoch, args, world_space=False, resolution=64, threshold=0.0):
    with open(args.json, 'r') as f:
        surface_configs = json.load(f)
    bound_min = torch.tensor(surface_configs['bbox_min'], dtype=torch.float32)
    bound_max = torch.tensor(surface_configs['bbox_max'], dtype=torch.float32)
    log.info('Coarse Bounding Box:')
    log.info([bound_min.numpy().tolist(), bound_max.numpy().tolist()])
    vertices, triangles =\
        extract_geometry(model, bound_min, bound_max, resolution=resolution, threshold=threshold)
    vertices = torch.from_numpy(vertices)
    v_min, v_max = vertices.amin(dim=0), vertices.amax(dim=0)
    vmin_ = (v_min - (v_max - v_min) * 0.1).clamp(bound_min , bound_max)
    vmax_ = (v_max + (v_max - v_min) * 0.1).clamp(bound_min, bound_max)
    log.info('Fine Bounding Box:')
    log.info([vmin_.numpy().tolist(), vmax_.numpy().tolist()])
    vertices, triangles =\
            extract_geometry(model, vmin_, vmax_, resolution=resolution, threshold=threshold)
    evals_folder_name = surface_configs['eval']
    os.makedirs(evals_folder_name, exist_ok=True)

    # if world_space:
      #   vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

    mesh = trimesh.Trimesh(vertices, triangles)
    mesh_save_path =os.path.join(evals_folder_name, 'N_{:0>8d}.ply'.format(epoch))
    mesh.export(mesh_save_path)
    log.info(f'Mesh saved in {mesh_save_path}')
    log.info('End')

def model_wrapper(p):
    with open(p, 'r') as f:
        surface_configs = json.load(f)
    log.info(f'Surface config loaded from {p}')
    evals_folder_name = surface_configs['eval']
    exps_folder_name = surface_configs['exp']
    utils.mkdir_ifnotexists(os.path.join('./', evals_folder_name))
    expdir = os.path.join('./', exps_folder_name)
    evaldir = os.path.join('./', evals_folder_name)
    utils.mkdir_ifnotexists(evaldir)
    iter = surface_configs['iteration']

    args = load_yaml(f'{expdir}/config.yaml')
    args.device_ids = [0]
    model, _, render_kwargs_train, render_kwargs_test, volume_render_fn = get_model(args)

    if torch.cuda.is_available():
        log.info('Cuda Detected')
        model.cuda()
    checkpoint_path = f'{expdir}/ckpts/{iter}.pt'

    # saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
    saved_model_state = torch.load(checkpoint_path)
    # print(saved_model_state.keys())
    
    model.load_state_dict(saved_model_state["model"])
    epoch = saved_model_state['global_step']    

    return model, epoch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--json', type=str, default='surface.json', help='Surface Configs.')
    opt = parser.parse_args()

    model, epoch = model_wrapper(opt.json)
    validate_mesh(model, epoch, opt, world_space=False, resolution=opt.resolution, threshold=0.0)