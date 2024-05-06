from models.base import ImplicitSurface, NeRF, RadianceNet
from utils import rend_util, train_util
from utils.logger import Logger

import copy
import functools
import numpy as np
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from nerfacc import accumulate_along_rays
from ..loss import polLoss
from ..PolAnalyser import normal_to_aop, visualize_aop

from ..cameras import world_to_camera, perspective_to_orthogonal, world_to_camera_orient, orthogonal_to_perspective, camera_projection, congruent_transform,\
                    create_step_vectors, indexing_2d_samples, create_step_vectors_2x,\
                    plot_normals, plot_vector_field, plot_pts, plot_pts_with_neighborhood

from ..math_utils import two_order_real_symmetric_matrix_svd

def cdf_Phi_s(x, s):
    return torch.sigmoid(x*s)


def sdf_to_alpha(sdf: torch.Tensor, s):
    # [(B), N_rays, N_pts]
    cdf = cdf_Phi_s(sdf, s)
    # [(B), N_rays, N_pts-1]
    # TODO: check sanity.
    opacity_alpha = (cdf[..., :-1] - cdf[..., 1:]) / (cdf[..., :-1] + 1e-10)
    opacity_alpha = torch.clamp_min(opacity_alpha, 0)
    return cdf, opacity_alpha


def sdf_to_w(sdf: torch.Tensor, s):
    device = sdf.device
    # [(B), N_rays, N_pts-1]
    cdf, opacity_alpha = sdf_to_alpha(sdf, s)

    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
            1.0 - opacity_alpha + 1e-10,
        ], dim=-1)
    
    # [(B), N_rays, N_pts-1]
    visibility_weights = opacity_alpha *\
        torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return cdf, opacity_alpha, visibility_weights


def alpha_to_w(alpha: torch.Tensor):
    device = alpha.device
    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*alpha.shape[:-1], 1], device=device),
            1.0 - alpha + 1e-10,
        ], dim=-1)
    
    # [(B), N_rays, N_pts-1]
    visibility_weights = alpha *\
        torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return visibility_weights

class NeuS(nn.Module):
    def __init__(self,
                 variance_init=0.05,
                 speed_factor=1.0,

                 input_ch=3,
                 W_geo_feat=-1,
                 use_outside_nerf=False,
                 obj_bounding_radius=1.0,

                 surface_cfg=dict(),
                 radiance_cfg=dict()):
        super().__init__()
        
        self.ln_s = nn.Parameter(data=torch.Tensor([-np.log(variance_init) / speed_factor]), requires_grad=True)
        self.speed_factor = speed_factor

        #------- surface network
        self.implicit_surface = ImplicitSurface(
            W_geo_feat=W_geo_feat, input_ch=input_ch, obj_bounding_size=obj_bounding_radius, **surface_cfg)
        
        #------- radiance network
        if W_geo_feat < 0:
            W_geo_feat = self.implicit_surface.W
        self.radiance_net = RadianceNet(
            W_geo_feat=W_geo_feat, **radiance_cfg)

        #-------- outside nerf++
        if use_outside_nerf:
            self.nerf_outside = NeRF(input_ch=4, multires=10, multires_view=4, use_view_dirs=True)

    def forward_radiance(self, x: torch.Tensor, view_dirs: torch.Tensor):
        _, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        radiance = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        return radiance

    def forward_s(self):
        return torch.exp(self.ln_s * self.speed_factor)

    def forward_surface(self, x: torch.Tensor):
        sdf = self.implicit_surface.forward(x)
        return sdf       

    def forward(self, x: torch.Tensor, view_dirs: torch.Tensor):
        sdf, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        radiances = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        return radiances, sdf, nablas


def volume_render(
    rays_o, 
    rays_d,
    model: NeuS,
    c2w = None,
    obj_bounding_radius=1.0,
    
    batched = False,
    batched_info = {},

    # render algorithm config
    calc_normal = True,
    use_view_dirs = True,
    rayschunk = 65536,
    netchunk = 1048576,
    white_bkgd = False,
    near_bypass: Optional[float] = None,
    far_bypass: Optional[float] = None,

    # render function config
    detailed_output = True,
    show_progress = False,

    # sampling related
    perturb = False,   # config whether do stratified sampling
    fixed_s_recp = 1/64.,
    N_samples = 64,
    N_importance = 64,
    N_outside = 0,  # whether to use outside nerf
    
    # upsample related
    upsample_algo = 'official_solution',
    N_nograd_samples = 2048,
    N_upsample_iters = 4,
    
    normal_splatting = False,
    normal_gaussian_estimate = False,
    cone_angle = None,
    gaussian_scale_factor = 1.0, 
    **dummy_kwargs  # just place holder
):
    """
    input: 
        rays_o: [(B,) N_rays, 3]
        rays_d: [(B,) N_rays, 3] NOTE: not normalized. contains info about ratio of len(this ray)/len(principle ray)
    """
    device = rays_o.device
    if batched:
        DIM_BATCHIFY = 1
        B = rays_d.shape[0]  # batch_size
        flat_vec_shape = [B, -1, 3]
    else:
        DIM_BATCHIFY = 0
        flat_vec_shape = [-1, 3]

    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()
    # NOTE: already normalized
    rays_d = F.normalize(rays_d, dim=-1)
    
    batchify_query = functools.partial(train_util.batchify_query, chunk=netchunk, dim_batchify=DIM_BATCHIFY)
    
    # c2w: [B, N, 4, 4] -> [B, 4, 4]
    c2w = c2w[:,0,:,:] 

    # ---------------
    # Render a ray chunk
    # ---------------
    def render_rayschunk(rays_o: torch.Tensor, rays_d: torch.Tensor):
        # rays_o: [(B), N_rays, 3]
        # rays_d: [(B), N_rays, 3]
        
        # [(B), N_rays] x 2
        near, far = rend_util.near_far_from_sphere(rays_o, rays_d, r=obj_bounding_radius)
        # NOTE: near = 0. will cause NaN in perspective projection transform, so clamping here
        near = near.clamp_min(0.01)
        if near_bypass is not None:
            near = near_bypass * torch.ones_like(near).to(device)
        if far_bypass is not None:
            far = far_bypass * torch.ones_like(far).to(device)
        
        if use_view_dirs:
            view_dirs = rays_d
        else:
            view_dirs = None
    
        prefix_batch = [B] if batched else []
        N_rays = rays_o.shape[-2]

        # ---------------
        # Sample points on the rays
        # ---------------


        # ---------------
        # Coarse Points
        
        # [(B), N_rays, N_samples]
        # d_coarse = torch.linspace(near, far, N_samples).float().to(device)
        # d_coarse = d_coarse.view([*[1]*len(prefix_batch), 1, N_samples]).repeat([*prefix_batch, N_rays, 1])
        _t = torch.linspace(0, 1, N_samples).float().to(device)
        d_coarse = near * (1 - _t) + far * _t
        
        # ---------------
        # Up Sampling
        with torch.no_grad():
            _d = d_coarse
            _sdf = batchify_query(model.implicit_surface.forward, rays_o.unsqueeze(-2) + _d.unsqueeze(-1) * rays_d.unsqueeze(-2))
            for i in range(N_upsample_iters):
                prev_sdf, next_sdf = _sdf[..., :-1], _sdf[..., 1:]
                prev_z_vals, next_z_vals = _d[..., :-1], _d[..., 1:]
                mid_sdf = (prev_sdf + next_sdf) * 0.5
                dot_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
                prev_dot_val = torch.cat([torch.zeros_like(dot_val[..., :1], device=device), dot_val[..., :-1]], dim=-1)   # jianfei: prev_slope, right shifted
                dot_val = torch.stack([prev_dot_val, dot_val], dim=-1)  # jianfei: concat prev_slope with slope
                dot_val, _ = torch.min(dot_val, dim=-1, keepdim=False)  # jianfei: find the minimum of prev_slope and current slope. (forward diff vs. backward diff., or the prev segment's slope vs. this segment's slope)
                dot_val = dot_val.clamp(-10.0, 0.0)
                
                dist = (next_z_vals - prev_z_vals)
                prev_esti_sdf = mid_sdf - dot_val * dist * 0.5
                next_esti_sdf = mid_sdf + dot_val * dist * 0.5
                
                prev_cdf = cdf_Phi_s(prev_esti_sdf, 64 * (2**i))
                next_cdf = cdf_Phi_s(next_esti_sdf, 64 * (2**i))
                alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
                _w = alpha_to_w(alpha)
                d_fine = rend_util.sample_pdf(_d, _w, N_importance // N_upsample_iters, det=not perturb)
                _d = torch.cat([_d, d_fine], dim=-1)
                
                sdf_fine = batchify_query(model.implicit_surface.forward, rays_o.unsqueeze(-2) + d_fine.unsqueeze(-1) * rays_d.unsqueeze(-2))
                _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
                _d, d_sort_indices = torch.sort(_d, dim=-1)
                _sdf = torch.gather(_sdf, DIM_BATCHIFY+1, d_sort_indices)
            d_all = _d # [B, N_rays, N_samples]
        # ------------------
        # Calculate Points
        # [(B), N_rays, N_samples+N_importance, 3]
        pts = rays_o[..., None, :] + rays_d[..., None, :] * d_all[..., :, None]
        # [(B), N_rays, N_pts-1, 3]
        # pts_mid = 0.5 * (pts[..., 1:, :] + pts[..., :-1, :])
        d_mid = 0.5 * (d_all[..., 1:] + d_all[..., :-1])
        pts_mid = rays_o[..., None, :] + rays_d[..., None, :] * d_mid[..., :, None]

        # ------------------
        # Inside Scene
        # ------------------
        # sdf, nablas, _ = model.implicit_surface.forward_with_nablas(pts)
        sdf, nablas, _ = batchify_query(model.implicit_surface.forward_with_nablas, pts)
        # [(B), N_ryas, N_pts], [(B), N_ryas, N_pts-1]
        cdf, opacity_alpha = sdf_to_alpha(sdf, model.forward_s()) # alpha_i is estimated by sdf_i and sdf_{i+1}
        # radiances = model.forward_radiance(pts_mid, view_dirs_mid)
        radiances = batchify_query(model.forward_radiance, pts_mid, view_dirs.unsqueeze(-2).expand_as(pts_mid) if use_view_dirs else None)

        # ------------------
        # Outside Scene
        # ------------------
        if N_outside > 0:
            _t = torch.linspace(0, 1, N_outside + 2)[..., 1:-1].float().to(device)
            d_vals_out = far / torch.flip(_t, dims=[-1])
            if perturb:
                _mids = .5 * (d_vals_out[..., 1:] + d_vals_out[..., :-1])
                _upper = torch.cat([_mids, d_vals_out[..., -1:]], -1)
                _lower = torch.cat([d_vals_out[..., :1], _mids], -1)
                _t_rand = torch.rand(_upper.shape).float().to(device)
                d_vals_out = _lower + (_upper - _lower) * _t_rand
            
            d_vals_out = torch.cat([d_mid, d_vals_out], dim=-1) # already sorted
            pts_out = rays_o[..., None, :] + rays_d[..., None, :] * d_vals_out[..., :, None]
            r = pts_out.norm(dim=-1, keepdim=True)
            x_out = torch.cat([pts_out/r, 1./r], dim=-1)
            views_out = view_dirs.unsqueeze(-2).expand_as(x_out[..., :3]) if use_view_dirs else None
            
            sigma_out, radiance_out = batchify_query(model.nerf_outside.forward, x_out, views_out)
            dists = d_vals_out[..., 1:] - d_vals_out[..., :-1]
            dists = torch.cat([dists, 1e10 * torch.ones(dists[..., :1].shape).to(device)], dim=-1)
            alpha_out = 1 - torch.exp(-F.softplus(sigma_out) * dists)   # use softplus instead of relu as NeuS's official repo
        
        # --------------
        # Ray Integration
        # --------------
        # [(B), N_rays, N_pts-1]
        if N_outside > 0:
            N_pts_1 = d_mid.shape[-1]
            # [(B), N_ryas, N_pts-1]
            mask_inside = (pts_mid.norm(dim=-1) <= obj_bounding_radius)
            # [(B), N_ryas, N_pts-1]
            alpha_in = opacity_alpha * mask_inside.float() + alpha_out[..., :N_pts_1] * (~mask_inside).float()
            # [(B), N_ryas, N_pts-1 + N_outside]
            opacity_alpha = torch.cat([alpha_in, alpha_out[..., N_pts_1:]], dim=-1)
            
            # [(B), N_ryas, N_pts-1, 3]
            radiance_in = radiances * mask_inside.float()[..., None] + radiance_out[..., :N_pts_1, :] * (~mask_inside).float()[..., None]
            # [(B), N_ryas, N_pts-1 + N_outside, 3]
            radiances = torch.cat([radiance_in, radiance_out[..., N_pts_1:, :]], dim=-2)
            d_final = d_vals_out
        else:
            d_final = d_mid

        # [(B), N_ryas, N_pts-1 + N_outside]
        visibility_weights = alpha_to_w(opacity_alpha)
        # [(B), N_rays]
        rgb_map = torch.sum(visibility_weights[..., None] * radiances, -2)
        # depth_map = torch.sum(visibility_weights * d_mid, -1)
        # NOTE: to get the correct depth map, the sum of weights must be 1!
        depth_map = torch.sum(visibility_weights / (visibility_weights.sum(-1, keepdim=True)+1e-10) * d_final, -1)
        acc_map = torch.sum(visibility_weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        ret_i = OrderedDict([
            ('rgb', rgb_map),           # [(B), N_rays, 3]
            ('depth_volume', depth_map),     # [(B), N_rays]
            # ('depth_surface', d_pred_out),    # [(B), N_rays]
            ('mask_volume', acc_map)            # [(B), N_rays]
        ])

        if calc_normal:
            normals_ = F.normalize(nablas, dim=-1)
            N_pts = min(visibility_weights.shape[-1], normals_.shape[-2])
            normals_map = (normals_[..., :N_pts, :] * visibility_weights[..., :N_pts, None]).sum(dim=-2)
            ret_i['normals_volume'] = F.normalize(normals_map, dim = -1)
            ret_i['normals_cam'] = camera_projection(normals_map, c2w)
 
        if normal_splatting:
            assert pts.shape[0] == 1, 'batch size must be one.'
            pts_ = pts[0,:,:-1] # N_ray, N_sample - 1, 3
            n_rays, n_samples = pts_.shape[0], pts_.shape[1]
            ray_indices = torch.arange(0,n_rays).type_as(pts_)[...,None].expand_as(pts_[:,:,0]).reshape(-1)
            pts_ = pts_.reshape(-1,3)
            pts_cam = world_to_camera(c2w[0,:3,:4], pts_)
            normal_cam = world_to_camera_orient(c2w[0,:,:], F.normalize(nablas[0,:,:-1], dim=-1)).reshape(-1,3)

            # Option A: Perspective Projection
            # normal_pts = pts_cam + 0.005 * F.normalize(normal_cam, -1)
            # pts_normal_ortho = perspective_to_orthogonal(normal_pts, near = _near, far = _far)
            # normal_ortho = F.normalize(pts_normal_ortho - pts_ortho, dim=-1)

            # Option B: Cross Product
            canonic_vec = torch.Tensor([0,0,-1])[None,...].type_as(normal_cam)
            view = F.normalize(pts_cam, dim=-1)
            normals_perspective = torch.linalg.cross(canonic_vec, torch.linalg.cross(view, normal_cam))
            comp_normals_perspective = accumulate_along_rays(visibility_weights[0,:].reshape(-1,1), ray_indices.long(), values=normals_perspective, n_rays=n_rays)
            comp_normals_ortho = accumulate_along_rays(visibility_weights[0,:].reshape(-1,1), ray_indices.long(), values=normal_cam, n_rays=n_rays)
            ret_i['normals_rayspace'] = comp_normals_perspective[None,...]
            ret_i['normals_ortho'] = comp_normals_ortho[None,...]

            if normal_gaussian_estimate: 
                _near = (-pts_cam[:,2]).min()
                _far = (-pts_cam[:,2]).max()
                _step_size = gaussian_scale_factor * _near / cone_angle 
                pts_ortho = perspective_to_orthogonal(pts_cam, near = _near, far = _far)
                step_vecs = create_step_vectors(pts_ortho)
                pts_upsamples_ortho = torch.stack([pts_ortho + _step_size * _vec.expand_as(pts_ortho) for _vec in step_vecs], -2)
                n_upsamples = len(step_vecs)
                pts_upsamples_world = orthogonal_to_perspective(pts_upsamples_ortho.reshape(-1,3), near= _near, far = _far).reshape(1,-1,n_upsamples,3)
                _,normals_upsample,_ = batchify_query(model.implicit_surface.forward_with_nablas, pts_upsamples_world)
                normals_conic = F.normalize(normals_upsample,dim = -1).reshape(1, n_rays, n_samples, n_upsamples, 3)
                normals_ = F.pad(normals_, (0,0,1,0), mode = 'replicate')
                gaussian_mean = normals_[...,1:-1,:] # [1, n_rays, n_samples, 3]
                gaussian_deviation_circle = (normals_conic.transpose(-1,-2) - gaussian_mean[...,None])
                gaussian_deviation_generatrix = ((normals_[...,:-2,:] - gaussian_mean) + (normals_[...,2:,:] - gaussian_mean))[...,None]
                gaussian_covariance = ((gaussian_deviation_circle @ gaussian_deviation_circle.transpose(-1,-2)) + (gaussian_deviation_generatrix @ gaussian_deviation_generatrix.transpose(-1,-2)))/(n_upsamples+1) 
                # ------------------
                # Gaussian Splatting
                # ------------------
                ## world-to-cam
                ## N_cam = R * N_world -> sigma' = R * sigma * R^T
                jacobian_world_to_cam = c2w[0,:3,:3]
                ## cam-to-perspective
                ## N_perspecive = Z_vec x (View x N_cam) -> sigma'' = V * sigma' * V^T, V = [[View_z, 0, -View_x],
                ##                                                                            [0,    View_z,-View_y],
                ##                                                                            [0,    0,    ,0    ]]
                jacobian_cam_to_perspective = torch.zeros_like(gaussian_covariance)
                view = view.reshape(1, n_rays, n_samples, 3)
                jacobian_cam_to_perspective[...,0,0] = view[...,2]
                jacobian_cam_to_perspective[...,1,1] = view[...,2]
                jacobian_cam_to_perspective[...,0,2] = -view[...,0]
                jacobian_cam_to_perspective[...,1,2] = -view[...,1]
                gaussian_cov = congruent_transform(congruent_transform(gaussian_covariance, jacobian_world_to_cam), jacobian_cam_to_perspective)[...,:2,:2]
                gaussian_cov_image = (gaussian_cov[..., :N_pts, :,:] * visibility_weights[..., :N_pts, None, None]).sum(dim=-3)
                ret_i['normals_image_covariance'] = gaussian_cov_image
                


        if detailed_output:
            ret_i['implicit_nablas'] = nablas
            ret_i['implicit_surface'] = sdf
            # ret_i['radiance'] = radiances
            # ret_i['alpha'] = opacity_alpha
            # ret_i['cdf'] = cdf
            # ret_i['visibility_weights'] = visibility_weights
            # ret_i['d_final'] = d_final
            if N_outside > 0:
                ret_i['sigma_out'] = sigma_out
                ret_i['radiance_out'] = radiance_out

        return ret_i
        
    ret = {}
    for i in tqdm(range(0, rays_o.shape[DIM_BATCHIFY], rayschunk), disable=not show_progress):
        ret_i = render_rayschunk(
            rays_o[:, i:i+rayschunk] if batched else rays_o[i:i+rayschunk],
            rays_d[:, i:i+rayschunk] if batched else rays_d[i:i+rayschunk]
        )
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_BATCHIFY)
    
    return ret['rgb'], ret['depth_volume'], ret

class SingleRenderer(nn.Module):
    def __init__(self, model: NeuS):
        super().__init__()
        self.model = model

    def forward(self, rays_o, rays_d, **kwargs):
        return volume_render(rays_o, rays_d, self.model, **kwargs)


class Trainer(nn.Module):
    def __init__(self, model: NeuS, device_ids=[0], batched=True):
        super().__init__()
        self.model = model
        self.renderer = SingleRenderer(model)
        if len(device_ids) > 1:
            self.renderer = nn.DataParallel(self.renderer, device_ids=device_ids, dim=1 if batched else 0)
        self.device = device_ids[0]
    
    def forward(self, 
                args,
                indices,
                model_input,
                ground_truth,
                render_kwargs_train: dict,
                it: int,
                device='cuda',
                ):

        intrinsics = model_input["intrinsics"].to(device)
        c2w = model_input['c2w'].to(device)
        H = render_kwargs_train['H']
        W = render_kwargs_train['W']
        rays_o, rays_d, select_inds = rend_util.get_rays(
            c2w, intrinsics, H, W, N_rays=args.data.N_rays, opengl= args.data.get('opengl', False))
        
        # get space step size of one pixel in the image
        render_kwargs_train['cone_angle'] = intrinsics[0,0,0].item() 

        # if it // 20 == 0:
        #     from models.ray_casting import plot_rays
        #     plot_rays(rays_o.reshape(-1,3), rays_d.reshape(-1,3), interval=50)
        # [B, N_rays, 3]
        target_rgb = torch.gather(ground_truth['rgb'].to(device), 1, torch.stack(3*[select_inds],-1))
        
        # NOTE: c2w [1, 4, 4] -> [1, 512, 4, 4] otherwise would be sliced
        rgb, depth_v, extras = self.renderer(rays_o, rays_d, c2w = c2w.expand([*rays_d.shape[:-1], 4, 4]), detailed_output=True, **render_kwargs_train) 
        # [B, N_rays, N_pts, 3]
        nablas: torch.Tensor = extras['implicit_nablas']
        # [B, N_rays, N_pts]
        nablas_norm = torch.norm(nablas, dim=-1)
        # [B, N_rays]
        mask_volume: torch.Tensor = extras['mask_volume']
        # NOTE: when predicted mask is close to 1 but GT is 0, exploding gradient.
        # mask_volume = torch.clamp(mask_volume, 1e-10, 1-1e-10)
        mask_volume = torch.clamp(mask_volume, 1e-3, 1-1e-3)
        extras['mask_volume_clipped'] = mask_volume

        losses = OrderedDict()
        normal_map = extras['normals_volume']
        normal_ray = extras['normals_rayspace']
        normal_ortho = extras['normals_ortho']
        normal_cam = extras['normals_cam']
        normals_image_cov = extras['normals_image_covariance']
        aop_map = torch.gather(ground_truth['AoP_map'].to(device), 1 , select_inds)
        dop_map = torch.gather(ground_truth['DoP_map'].to(device), 1 , select_inds)
        aop_sample_idx = indexing_2d_samples(select_inds, H, W, args.model.get('gaussian_scale_factor', 1.0)).reshape(1,-1) # [1 n_rays, 4] -> [1, n_rays * 4]
        aop_samples = torch.gather(ground_truth['AoP_map'].to(device), 1 , aop_sample_idx.long()).reshape(*aop_map.shape, 4)
        dop_samples = torch.gather(ground_truth['DoP_map'].to(device), 1 , aop_sample_idx.long()).reshape(*aop_map.shape, 4)
        
        c2w = model_input['c2w'].to(device)
        
        if 'mask' in ground_truth:
            target_mask = torch.gather(ground_truth["mask"].to(device), 1, select_inds)
        else:
            target_mask = None

        if 'mask_ignore' in ground_truth:
            mask_ignore = torch.gather(ground_truth["mask_ignore"].to(device), 1, select_inds)
        else:
            mask_ignore = None

        pLoss = polLoss(args.training.loss)
        pred = OrderedDict([
            ('rgb',rgb),
            ('normal_map',normal_map),
            ('grad_norm', nablas_norm),
            ('mask_volume', mask_volume),
            ('normals_rayspace', normal_ray),
            ('normals_ortho', normal_ortho),
            ('normals_image_cov', normals_image_cov)
            # ('normals_cam', normal_cam)
        ])
        gt = OrderedDict([
            ('rgb',target_rgb),
            ('AoP_map', aop_map),
            ('DoP_map', dop_map),
            ('mask', target_mask),
            ('c2w', c2w),
            ('aop_samples', aop_samples),
            ('dop_samples', dop_samples),
            ('mask_ignore', mask_ignore)
        ])
        losses, _ = pLoss(pred, gt, it)
        extras['implicit_nablas_norm'] = nablas_norm  
        extras['scalars'] = {'1/s': 1./self.model.forward_s().data}
        extras['select_inds'] = select_inds          
        return OrderedDict(
            [('losses', losses),
            ('extras', extras)])
    
    def val_pol(self,
                logger:Logger, 
                ret, # extras{} from SingleRender
                c2w, # pose
                gt, # PolData.gt.items
                to_img, 
                it, 
                render_kwargs_test):
        
        # [1, N_rays, 1] -> [H, W]
        aop_map = to_img(gt['AoP_map'][...,None])[0,0]
        dop_map = to_img(gt['DoP_map'][...,None])[0,0]
        aop_sat_rgb = visualize_aop(aop_map, dop_map).permute(2,0,1) # C, H, W
        logger.add_imgs(aop_sat_rgb, 'val/gt_aop', it)

        aop_rgb = visualize_aop(aop_map).permute(2,0,1) 
        logger.add_imgs(aop_sat_rgb, 'val/gt_aop_ori', it)
        # normal_ortho = ret['normals_ortho']
        # pred_aop = to_img(normal_to_aop(normal_ortho, render_kwargs_test['opengl'])[...,None])[0,0]
        # pred_aop_ortho_rgb = visualize_aop(pred_aop, dop_map).permute(2,0,1) 
        # logger.add_imgs(pred_aop_ortho_rgb, 'val/pred_aop_ortho', it)

        # aop_error_map = torch.abs(pred_aop - aop_map)/np.pi
        # logger.add_imgs(aop_error_map, 'val/aop_error', it)

        normal_ray = ret['normals_rayspace']
        pred_aop_ray = to_img(normal_to_aop(normal_ray, render_kwargs_test['opengl'])[...,None])[0,0]
        pred_aop_dop_rgb = visualize_aop(pred_aop_ray, dop_map).permute(2,0,1) # C, H, W
        logger.add_imgs(pred_aop_dop_rgb, 'val/pred_aop_ray', it)

        cov_pred = ret['normals_image_covariance']
        singular_pred = torch.linalg.svdvals(cov_pred)
        anistropic_pred = (singular_pred[...,0] -  singular_pred[...,1]).abs()
        v_rgb = to_img((F.normalize(ret['normals_volume'], dim = -1)/2.+0.5))
        v_alpha = to_img(anistropic_pred[...,None]/anistropic_pred.max())
        v_normal_alpha = torch.concat([v_rgb, v_alpha], dim=-3)
        v_alpha_ = v_alpha.expand_as(v_normal_alpha).clone() # if need inplace op
        v_alpha_[:,3,:,:] = 1.0


        # rgb -> rgba
        mask = to_img(gt['mask'][...,None])
        # mask_ignore = to_img(gt['mask_ignore'][...,None].float())
        # logger.add_imgs(mask_ignore, 'val/mask_ignore', it)
        target_img = to_img(gt['rgb'])
        target_img = torch.concat([target_img, mask], dim=-3)
        v_normal = torch.concat([v_rgb, mask], dim=-3)
        normal_img_alpha = v_normal_alpha
        pred_aop_rgb = torch.cat([visualize_aop(pred_aop_ray).permute(2,0,1).type_as(v_normal_alpha)[None,...], mask], dim=1)
        pred_aop_dop_rgb = torch.cat([pred_aop_dop_rgb.type_as(v_alpha)[None,...], mask], dim=1)

        # GT Covariance
        normals_aop_samples=torch.stack([torch.sin(gt['aop_samples']), -torch.cos(gt['aop_samples'])], dim = -1)
        normals_aop_mean = torch.stack([torch.sin(gt['AoP_map']), -torch.cos(gt['AoP_map'])], dim = -1)
        normals_aop_samples = (normals_aop_samples - normals_aop_mean[...,None,:])
        normals_aop_cov = (normals_aop_samples.transpose(-1,-2) @ normals_aop_samples)/3
        aop_svd_val, aop_svd_vec = two_order_real_symmetric_matrix_svd(normals_aop_cov)
        anistropic_aop = (aop_svd_val[...,1] /  (aop_svd_val[...,0] + 1e-8))
        anistropic_weight = torch.ones_like(anistropic_aop)
        principle_angle = torch.atan2(aop_svd_vec[...,1], aop_svd_vec[...,0])
        if mask.sum()>0:
            anistropic_weight[gt['mask']] = anistropic_aop[gt['mask']]
        # anistropic_weight = (-anistropic_weight.log()).clamp(0,100)
        anistropic_weight /= anistropic_weight.max()
        anistropic_img = TF.equalize((to_img(anistropic_weight[...,None])*255).type(torch.uint8))/255
        principle_angle= to_img(principle_angle)[0,0]
        principle_angle_img = visualize_aop(principle_angle).permute(2,0,1)
        principle_angle_img_rew = visualize_aop(principle_angle, anistropic_img[0,0]).permute(2,0,1)
        principle_angle_img[~(mask[0,...].expand_as(principle_angle_img))] = 1.0
        principle_angle_img_rew[~(mask[0,...].expand_as(principle_angle_img))] = 1.0
        # available_pts = anistropic_img[0,0]>0.5
        # principle_angle_img_rew_npy = ((principle_angle_img_rew.permute(1,2,0)*255).type(torch.uint8)).numpy()
        # available_pts_dir = principle_angle[available_pts.cpu()].cpu().numpy()
        # available_pts_scale = anistropic_img[0,0][available_pts].cpu().numpy()
        # available_pts_grid = np.meshgrid(np.arange(0,principle_angle.shape[0]), np.arange(0,principle_angle.shape[1]), indexing='ij')
        # available_pts_x = available_pts_grid[0][available_pts.cpu().numpy()]
        # available_pts_y = available_pts_grid[1][available_pts.cpu().numpy()]

        logger.add_imgs(anistropic_img,'val/anisotropy', it)
        logger.add_imgs(principle_angle_img,'val/pd', it)
        logger.add_imgs(principle_angle_img_rew,'val/pd_rew', it)
        logger.add_imgs(torch.cat([target_img, v_normal, normal_img_alpha, pred_aop_rgb, 
                                   pred_aop_dop_rgb, v_alpha_],dim=0), 'val/pred_normal_cov', it)


def get_model(args):
    
    if not args.training.with_mask:
        assert 'N_outside' in args.model.keys() and args.model.N_outside > 0, \
            "Please specify a positive model:N_outside for neus with nerf++"
    
    model_config = {
        'obj_bounding_radius':  args.model.obj_bounding_radius,
        'W_geo_feat':       args.model.setdefault('W_geometry_feature', 256),
        'use_outside_nerf': not args.training.with_mask,
        'speed_factor':     args.training.setdefault('speed_factor', 1.0),
        'variance_init':    args.model.setdefault('variance_init', 0.05)
    }
    
    surface_cfg = {
        'use_siren':    args.model.surface.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.surface.setdefault('embed_multires', 6),
        'radius_init':  args.model.surface.setdefault('radius_init', 1.0),
        'geometric_init': args.model.surface.setdefault('geometric_init', True),
        'D': args.model.surface.setdefault('D', 8),
        'W': args.model.surface.setdefault('W', 256),
        'skips': args.model.surface.setdefault('skips', [4]),
    }
        
    radiance_cfg = {
        'use_siren':    args.model.radiance.setdefault('use_siren', args.model.setdefault('use_siren', False)),
        'embed_multires': args.model.radiance.setdefault('embed_multires', -1),
        'embed_multires_view': args.model.radiance.setdefault('embed_multires_view', -1),
        'use_view_dirs': args.model.radiance.setdefault('use_view_dirs', True),
        'D': args.model.radiance.setdefault('D', 4),
        'W': args.model.radiance.setdefault('W', 256),
        'skips': args.model.radiance.setdefault('skips', []),
    }
    
    model_config['surface_cfg'] = surface_cfg
    model_config['radiance_cfg'] = radiance_cfg
    
    model = NeuS(**model_config)
    
    ## render kwargs
    render_kwargs_train = {
        # upsample config
        'upsample_algo':    args.model.setdefault('upsample_algo', 'official_solution'),    # [official_solution, direct_more, direct_use]
        'N_nograd_samples': args.model.setdefault('N_nograd_samples', 2048),
        'N_upsample_iters': args.model.setdefault('N_upsample_iters', 4), 
        'N_outside': args.model.setdefault('N_outside', 0),
        'obj_bounding_radius': args.data.setdefault('obj_bounding_radius', 1.0),
        'batched': args.data.batch_size is not None,
        'perturb': args.model.setdefault('perturb', True),   # config whether do stratified sampling
        'white_bkgd': args.model.setdefault('white_bkgd', False),
        'normal_splatting': args.model.setdefault('normal_splatting', True),
        'normal_gaussian_estimate': args.model.setdefault('normal_gaussian_estimate', True),
        'gaussian_scale_factor': args.model.setdefault('gaussian_scale_factor', 1.0),
        'has_pol': True
        
    }
    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test['rayschunk'] = args.data.val_rayschunk
    render_kwargs_test['perturb'] = False
    render_kwargs_test['opengl'] = args.data.get('opengl', False)
    trainer = Trainer(model, device_ids=args.device_ids, batched=render_kwargs_train['batched'])
    
    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer
