'''From Polarized VolSDF, define AoP Loss
'''
import torch
from torch import nn
import utils.general as utils

import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from .math_utils import two_order_real_symmetric_matrix_svd

def covariance_to_correlation(cov):
    """2d covariance -> correlation
        sigma(x_1, x_2)/(sigma(x_1)sigma(x_2))
    Args:
        cov: _, 2, 2
    """
    assert cov.shape[-2] == cov.shape[-1] and cov.shape[-1] == 2  
    print('cov', cov.abs().min())
    std_x, std_y = (cov[...,0,0] + 1e-7).sqrt(), (cov[...,1,1]+ 1e-7).sqrt()
    # cov[...,0,0] /= std_x * std_x
    cov[...,0,0] = 1.
    cov[...,0,1] /= (std_x * std_y)
    cov[...,1,0] /= (std_y * std_x)
    print('divisor', (std_x * std_y).min())
    # cov[...,1,1] /= std_y * std_y
    cov[...,1,1] = 1.
    return cov


class polLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w = args
        
    def forward(self, model_outputs, ground_truth, iteration):

        rgb_pred =  model_outputs['rgb']
        rgb_gt = ground_truth['rgb']
        AoP_gt = ground_truth['AoP_map']
        DoP_gt = ground_truth['DoP_map']
        mask = ground_truth['mask']
        grad_norm = model_outputs['grad_norm']
        mask_ignore = ground_truth['mask_ignore']
        normal_map_cam = model_outputs['normals_rayspace'] if self.w.normal_perspective else model_outputs['normals_ortho']

        losses = OrderedDict()

        losses['loss_img'] = F.l1_loss(rgb_pred, rgb_gt, reduction='none')
        losses['loss_img'] = self.w.w_rgb * losses['loss_img']

        if self.w.pol_rew and iteration > self.w.pol_start_it:
            dop_w = torch.clamp(DoP_gt, min = 0.0, max = self.w.dop_upper) if self.w.dop_upper > 0 else DoP_gt
            losses['loss_img'] = (1-dop_w[...,None]) * losses['loss_img']
        if self.w.w_mask > 0:
            losses['loss_img'] = (losses['loss_img'] * mask[..., None].float()).sum() / (mask.sum() + 1e-10)
            losses['loss_mask'] = self.w.w_mask * F.binary_cross_entropy(model_outputs['mask_volume'], mask.float(), reduction='mean')
        else:
            losses['loss_img'] = losses['loss_img'].mean()

        if self.w.w_eik > 0:
            losses['loss_eikonal'] = self.w.w_eik * F.mse_loss(grad_norm, grad_norm.new_ones(grad_norm.shape), reduction='mean')

        if self.w.w_aop > 0 and iteration > self.w.pol_start_it:
            azi_angle = torch.atan2(normal_map_cam[...,1], normal_map_cam[...,0] + 1e-10) # N_batch x N_rays  (rad) [-pi,pi]
            aop_pred = torch.remainder(np.pi/2 + azi_angle, np.pi) if self.w.opengl else torch.remainder(np.pi/2 - azi_angle, np.pi) 
            eta = F.l1_loss(aop_pred, AoP_gt, reduction='none')
            _mask = mask * (~mask_ignore) if mask_ignore is not None else mask
            if self.w.pol_rew:
                AoP_loss = (DoP_gt * _mask.float() * eta).sum()/ (_mask.sum() + 1e-10) if  self.w.aop_mask else (DoP_gt * eta).mean()  
            else:
                AoP_loss = (_mask.float() * eta).sum()/ (_mask.sum() + 1e-10) if  self.w.aop_mask else (DoP_gt * eta).mean()              
            losses['loss_aop'] = self.w.w_aop * AoP_loss
            
        if self.w.w_splat > 0 and iteration > self.w.splat_start_it:
            _norm_scale = normal_map_cam.norm(dim = -1)[...,None]
            normals_aop_mean = _norm_scale * torch.stack([torch.sin(AoP_gt), -torch.cos(AoP_gt)], dim = -1)
            normals_aop_samples=_norm_scale[...,None] * torch.stack([torch.sin(ground_truth['aop_samples']), -torch.cos(ground_truth['aop_samples'])], dim = -1)
            normals_dop_samples= ground_truth['dop_samples']
            normals_aop_samples = (normals_aop_samples - normals_aop_mean[...,None,:])
            normals_aop_cov = (normals_aop_samples.transpose(-1,-2) @ normals_aop_samples)/3
            normals_image_cov = model_outputs['normals_image_cov']
            if self.w.get('svd_sup', False):
            #----- test: SVD Sup. --------
                # NOTE: linalg.svd Cause backward NaN
                # img_svd_vec = torch.linalg.svd(normals_image_cov , driver = "gesvd" )[0]
                # img_svd_val = torch.linalg.svd(normals_image_cov, driver = "gesvd" )[1]
                # aop_svd_vec = torch.linalg.svd(normals_aop_cov, driver = "gesvd")[0]
                img_svd_val, img_svd_vec = two_order_real_symmetric_matrix_svd(normals_image_cov)
                aop_svd_val, aop_svd_vec = two_order_real_symmetric_matrix_svd(normals_aop_cov)
                anistropic_img = (img_svd_val[...,1] / (img_svd_val[...,0] + 1e-8)) * mask.float()
                anistropic_aop = (aop_svd_val[...,1] /  (aop_svd_val[...,0] + 1e-8)) * mask.float()
                anistropic_weight = torch.ones_like(anistropic_aop)
                scale_factor = 1/10
                anistropic_weight *= scale_factor
                vec_orientation_similarity = F.cosine_similarity(img_svd_vec, aop_svd_vec, dim=-1).abs().sum(dim=-1)/2
                eta = F.l1_loss(anistropic_img, anistropic_aop, reduction='none')* mask.float()
                eta += F.l1_loss(vec_orientation_similarity, torch.ones_like(vec_orientation_similarity), reduction = 'none') * mask.float() * anistropic_weight
            else:
                eta = F.l1_loss(normals_image_cov, normals_aop_cov, reduction='none')* mask.float()
            if self.w.splat_rew:
                dop_w = normals_dop_samples.mean(dim=-1)
                eta *= dop_w
            losses['loss_gauss'] = self.w.w_splat * eta.sum() / (mask.sum() + 1e-10)

        loss = 0
        for k, v in losses.items():
            loss += losses[k]
        losses['total'] = loss
        return losses, None
