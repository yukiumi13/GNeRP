import matplotlib.pyplot as plt
import torch
from models.frameworks import get_model
from utils import rend_util, train_util, mesh_util, io_util
import os, json, random
import numpy as np

data_dir = 'data/pol/ceramicCat'
idx = 1
H = 2048
W=2448

config_path_1 = 'logs/VolSDF/baseline/ceramicCat/config.yaml'
config_path_2 = 'logs/PNeuS/ceramicCat/wrgb_0.1/config.yaml'


ckpt_path_1 = os.path.join(*config_path_1.split('/')[:-1],'ckpts/latest.pt')
print(ckpt_path_1)

ckpt_path_2 = os.path.join(*config_path_2.split('/')[:-1],'ckpts/latest.pt')
print(ckpt_path_2)

config_1 = io_util.load_yaml(config_path_1)
config_1.device_ids = [0]
config_2 = io_util.load_yaml(config_path_2)
config_2.device_ids = [0]
model_1, trainer_1, render_kwargs_train_1, render_kwargs_test_1, volume_render_fn_1 = get_model(config_1)
model_1.cuda()
model_1.eval()
model_2, trainer_2, render_kwargs_train_2, render_kwargs_test_2, volume_render_fn_2 = get_model(config_2)
model_2.cuda()
model_2.eval()

state_dict = torch.load(ckpt_path_1)
model_1.load_state_dict(state_dict['model'])
state_dict = torch.load(ckpt_path_2)
model_2.load_state_dict(state_dict['model'])

with open(os.path.join(data_dir, 'camera_intrinsics.json'), 'r') as f:
        camera_intrinsics = json.load(f)['intrinsics']

with open(os.path.join(data_dir, 'camera_extrinsics.json'), 'r') as f:
        camera_extrinsics = json.load(f)

name_idx = list(camera_extrinsics.keys())[idx]
print(name_idx)
rotation = torch.Tensor(camera_extrinsics[name_idx]['rotation']).transpose(1, 0).float() # R^T
translation = torch.Tensor(camera_extrinsics[name_idx]['camera_pos']).float() # C = -R_transpose*t
c2w_ = torch.cat([rotation, translation.unsqueeze(1)], dim=1) # 3 x 4
c2w = torch.cat([c2w_, torch.Tensor([[0.,0.,0.,1.]])], dim=0)
cam_center_norms = np.linalg.norm(translation.numpy())
c2w_1 = c2w * (3.0 / cam_center_norms)
c2w_2 = c2w * (2.0 / cam_center_norms)
intrinsics = torch.Tensor(camera_intrinsics)
rays_o_1, rays_d_1, rays_o_2, rays_d_2, select_inds = rend_util.get_birays(
            c2w_1, c2w_2, intrinsics, H, W, N_rays=512)
rays_o_1 = rays_o_1.cuda()[None,...]
rays_d_1 = rays_d_1.cuda()[None,...]
rays_o_2 = rays_o_2.cuda()[None,...]
rays_d_2 = rays_d_2.cuda()[None,...]

rgb, depth_v, extras_1 = volume_render_fn_1(rays_o_1, rays_d_1, detailed_output=True, **render_kwargs_train_1)
rgb, depth_v, extras_2 = volume_render_fn_2(rays_o_2, rays_d_2, detailed_output=True, **render_kwargs_train_2)

fig = plt.figure()
alphas_1 = extras_1['alpha']
sdfs_1 = extras_1['implicit_surface']
weights_1 = extras_1['visibility_weights']
try:
    cdfs_1 = extras_1['cdf']
except:
    cdfs_1 = extras_1['sigma']
try:
    depths_1 = extras_1['d_final']
except:
    depths_1 = extras_1['d_vals']
try:
    beta_1 = extras_1['beta_warp']
except:
    beta_1 = None

alphas_2 = extras_2['alpha']
sdfs_2 = extras_2['implicit_surface']
weights_2 = extras_2['visibility_weights']
try:
    cdfs_2 = extras_2['cdf']
except:
    cdfs_2 = extras_2['sigma']
try:
    depths_2 = extras_2['d_final']
except:
    depths_2 = extras_2['d_vals']
try:
    beta_2 = extras_2['beta_warp']
except:
    beta_2 = None

N = sdfs_1.shape[-1] - 1

ind_1 = random.randint(0, N)
#depths = torch.arange(N + 32)

alpha_1 = alphas_1[0][ind_1][:127].detach().cpu().numpy()
alpha_1 = np.concatenate([alpha_1, np.ones(1)], axis=-1)
weight_1 = weights_1[0][ind_1][:127].detach().cpu().numpy()
weight_1 = np.concatenate([weight_1, np.zeros(1)], axis=-1)
sdf_1 = sdfs_1[0][ind_1][:127].detach().cpu().numpy()
#cdf_1 = torch.cat((cdfs[0][ind_1], cdfs[0][ind_1][-1] *torch.ones((31)).cuda()), dim=-1)
cdf_1 = cdfs_1[0][ind_1][:127].detach().cpu().numpy()
depths_1 = depths_1[0][ind_1][:127].detach().cpu().numpy()

alpha_2 = alphas_2[0][ind_1][:127].detach().cpu().numpy()
weight_2 = weights_2[0][ind_1][:127].detach().cpu().numpy()
sdf_2 = sdfs_2[0][ind_1][:127].detach().cpu().numpy()
#cdf_1 = torch.cat((cdfs[0][ind_1], cdfs[0][ind_1][-1] *torch.ones((31)).cuda()), dim=-1)
cdf_2 = cdfs_2[0][ind_1][:127].detach().cpu().numpy()
depths_2 = depths_2[0][ind_1][:127].detach().cpu().numpy()

if beta_2 is not None:
    beta_2 = beta_2[0][ind_1][:127].detach().cpu().numpy()

fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
# ax[0, 0].scatter(x=depths, y=sdf_1)
# ax[0, 0].set_xlabel('Depth', fontsize=16)
# ax[0, 0].set_ylabel('sdf', fontsize=16)
# ax[0, 0].set_title('sdf', fontsize=16)

# if beta is not None:
#     ax[0, 1].scatter(x=depths, y=beta_1)
#     ax[0, 1].set_xlabel('Depth', fontsize=16)
#     ax[0, 1].set_ylabel('beta_warp', fontsize=16)
#     ax[0, 1].set_title('beta_warp', fontsize=16)
# else:
#     ax[0, 1].scatter(x=depths, y=cdf_1)
#     ax[0, 1].set_xlabel('Depth', fontsize=16)
#     ax[0, 1].set_ylabel('cdf', fontsize=16)
#     ax[0, 1].set_title('cdf', fontsize=16)

# ax[1, 0].scatter(x=depths, y=alpha_1)
# ax[1, 0].set_xlabel('Depth', fontsize=16)
# ax[1, 0].set_ylabel('alpha', fontsize=16)
# ax[1, 0].set_title('alpha', fontsize=16)

ax[0].scatter(x=depths_1, y=weight_1)
ax[0].set_xlabel('Depth', fontsize=16)
ax[0].set_xticks([])
ax[0].set_ylabel('weight', fontsize=16)
ax[0].set_ylim(0,0.5)
ax[0].set_title('VolSDF', fontsize=16)
ax[1].scatter(x=depths_2, y=weight_2)
ax[1].set_xlabel('Depth', fontsize=16)
ax[1].set_xticks([])
ax[1].set_ylabel('weight', fontsize=16)
ax[1].set_ylim(0,0.5)
ax[1].set_title('PNeuS', fontsize=16)


fig.savefig('./weights_vis.png')
