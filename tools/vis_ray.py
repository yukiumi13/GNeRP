import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
sys.path.append('/dataset/yokoli/neurecon')
from utils.rend_util import get_rays
# from dataio.normalData import SceneDataset
from dataio.PolData import SceneDataset
import torch.nn.functional as F
def lift(x, y, z, intrinsics):
    device = x.device
    # parse intrinsics
    intrinsics = intrinsics.to(device)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    sk = intrinsics[..., 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(device)), dim=-1)
def lift_opengl(x, y, z, intrinsics):
    # NOTE: OpenGL convention
    device = x.device
    # parse intrinsics
    intrinsics = intrinsics.to(device)
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]

    x_lift = (x - cx.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous and CONVERT TO OPENGL
    return torch.stack((x_lift, -y_lift, -z, torch.ones_like(z).to(device)), dim=-1)

def get_center_ray(c2w, intrinsics, H, W, N_rays=1):
    device = c2w.device
    cam_loc = c2w[..., :3, 3]
    p = c2w

    prefix = p.shape[:-2]
    device = c2w.device
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t().to(device).reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])
    j = j.t().to(device).reshape([*[1]*len(prefix), H*W]).expand([*prefix, H*W])

    if N_rays > 0:
        N_rays = min(N_rays, H*W)
        # ---------- option 1: full image uniformly randomize
        # select_inds = torch.from_numpy(
        #     np.random.choice(H*W, size=[*prefix, N_rays], replace=False)).to(device)
        # select_inds = torch.randint(0, H*W, size=[N_rays]).expand([*prefix, N_rays]).to(device)
        # ---------- option 2: H/W seperately randomize
        select_hs = torch.Tensor([H//2]).long()
        select_ws = torch.Tensor([W//2]).long()
        select_inds = select_hs * W + select_ws
        select_inds = select_inds.expand([*prefix, N_rays])

        i = torch.gather(i, -1, select_inds)
        j = torch.gather(j, -1, select_inds)

    # pixel_points_cam = lift(i, j, torch.ones_like(i).to(device), intrinsics=intrinsics)

    pixel_points_cam = lift_opengl(i, j, torch.ones_like(i).to(device), intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.transpose(-1,-2)

    # NOTE: left-multiply.
    #       after the above permute(), shapes of coordinates changed from [B,N,4] to [B,4,N], which ensures correct left-multiplication
    #       p is camera 2 world matrix.
    if len(prefix) > 0:
        world_coords = torch.bmm(p, pixel_points_cam).transpose(-1, -2)[..., :3]
    else:
        world_coords = torch.mm(p, pixel_points_cam).transpose(-1, -2)[..., :3]
    rays_d = world_coords - cam_loc[..., None, :]
    rays_d = F.normalize(rays_d, dim=2)

    rays_o = cam_loc[..., None, :].expand_as(rays_d)

    return rays_o, rays_d, select_inds

def plot_rays(rays_o: np.ndarray, rays_d: np.ndarray, ax):
    # TODO: automatic reducing number of rays
    XYZUVW = np.concatenate([rays_o, rays_d], axis=-1)
    X, Y, Z, U, V, W = np.transpose(XYZUVW)
    # X2 = X+U
    # Y2 = Y+V
    # Z2 = Z+W
    # x_max = max(np.max(X), np.max(X2))
    # x_min = min(np.min(X), np.min(X2))
    # y_max = max(np.max(Y), np.max(Y2))
    # y_min = min(np.min(Y), np.min(Y2))
    # z_max = max(np.max(Z), np.max(Z2))
    # z_min = min(np.min(Z), np.min(Z2))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W)
    # ax.set_xlim(x_min, x_max)
    # ax.set_ylim(y_min, y_max)
    # ax.set_zlim(z_min, z_max)
    
    return ax

dataset = SceneDataset(False, '/dataset/yokoli/data/pol/mitsuba_bunny', downscale=32, scale_radius = 3, chromatic='sRGB',opengl=True)

fig = plt.figure(figsize=[19.2,10.8])
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
H, W = (dataset.H, dataset.W)

for i in range(dataset.n_images):
    _, model_input, _ = dataset[i]
    intrinsics = model_input["intrinsics"][None, ...]
    c2w = model_input['c2w'][None, ...]
    # c2w = dataset.get_gt_pose(scaled=True)
    rays_o, rays_d, select_inds = get_center_ray(c2w, intrinsics, H, W, N_rays=1)
    rays_o = rays_o.data.squeeze(0).cpu().numpy()
    rays_d = rays_d.data.squeeze(0).cpu().numpy()
    # x y z -> x z -y
    # rays_o = rays_o[:,[0,2,1]]
    # rays_d = rays_d[:,[0,2,1]]
    # rays_o[:,2] = -rays_o[:,2]
    # rays_d[:,2] = -rays_d[:,2]
    ax = plot_rays(rays_o, rays_d, ax)
fig.savefig('rays.png', bbox_inches='tight')