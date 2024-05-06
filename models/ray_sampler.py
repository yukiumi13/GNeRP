from models.base import ImplicitSurface, NeRF, RadianceNetMono
from utils import io_util, train_util, rend_util
from utils.logger import Logger

import copy
import functools
import matplotlib.pyplot as plt
from collections import OrderedDict
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F



class RaySampler(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_z_vals(self):
        pass

class UniformSampler(RaySampler):
    def __init__(self, N_samples):
        super().__init__()  
        self.N_samples = N_samples

    def get_z_vals(self, nears, fars, perturb, device):
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
        z_vals = nears * (1. - t_vals) + fars * (t_vals)

        if perturb:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(device)

            z_vals = lower + (upper - lower) * t_rand

        return z_vals

class ErrorBoundSampler(RaySampler):
    """
    @ Section 3.4 in the paper.
    Args:
        implicit_surface_fn. sdf query function.
        init_dvals: [..., N_rays, N]
        rays_o:     [..., N_rays, 3]
        rays_d:     [..., N_rays, 3]
    Return:
        final_fine_dvals:   [..., N_rays, final_N_importance]
        beta:               [..., N_rays]. beta heat map
    """
    # NOTE: this algorithm is parallelized for every ray!!!
    def __init__(self,
                implicit_surface_fn, # ImplicitNetwork                 
                eps=0.1, 
                max_iter:int=5, 
                max_bisection:int=10, 
                final_N_importance:int=64,
                N_init:int=128, 
                N_up:int=128,
                N_extra:int=32,
                perturb:bool=True # Stratified Sampling in Uniform Initialization
            ):
        super().__init__()
        self.implicit_surface_fn = implicit_surface_fn
        self.eps = eps
        self.max_iter = max_iter
        self.max_bisection = max_bisection
        self.N_fin = final_N_importance
        self.N_extra = N_extra
        self.N_up = N_up
        self.perturb = perturb
        self.uniform_sampler = UniformSampler(N_init)
    
    @torch.no_grad()
    def query_sdf(self, d_vals_, samples_idx, rays_o_, rays_d_, sdf):
        pts = rays_o_[..., None, :] + rays_d_[..., None, :] * d_vals_[..., :, None]
        samples_sdf = self.implicit_surface_fn(pts)
        sdf_merge = torch.cat([sdf, samples_sdf],-1)
        sdf = torch.gather(sdf_merge, -1, samples_idx)
        return sdf
    
    def sdf_to_sigma(self, sdf: torch.Tensor, alpha, beta):
        return alpha.unsqueeze(-1) * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta.unsqueeze(-1)))

    def error_bound(self, beta, sdf, deltas, d_star):
        device = deltas.device
        sigma = self.sdf_to_sigma(sdf, 1/beta, beta)
        # \hat{R}(t_n)
        left_section_est = torch.cat([torch.zeros([*deltas.shape[:-1],1], device=device), deltas*sigma[...,:-1]], dim=-1)
        left_riemann_sum = torch.cumsum(left_section_est, dim=-1)
        # \hat{E}(t_n)
        error_i = torch.exp(-d_star / beta.unsqueeze(-1)) * (deltas ** 2.) / (4 * beta.unsqueeze(-1) ** 2) # [..., N-1]
        error_sum = torch.cumsum(error_i, dim=-1)
        error_bound_i = (torch.clamp(torch.exp(error_sum), max=1.e6) - 1.0) * torch.exp(-left_riemann_sum[..., :-1])

        return error_bound_i.max(-1)[0]

    def get_z_vals(self, 
                rays_o, # [..., N_rays, 3]
                rays_d, # [..., N_rays, 3]
                nears, # [..., N_rays, 1]
                fars, # [..., N_rays, 1]
                alpha_net, #[..., N_rays] 
                beta_net, # [...,N_rays]
                device,
                debug:bool=True):

        prefix = rays_o.shape[:-1] # [(B), N_rays]
        beta0 = beta_net.detach()
        
        if debug:
            print(f'beta0 from ImplicitNet:', beta0)
            print('nears:',nears.shape, 'fars', fars.shape)

        # NOTE: delta_i = d_vals_{i+1} - d_vals_{i}

        # Initial Uniform Sampling
        d_vals = self.uniform_sampler.get_z_vals(nears, fars, self.perturb, device)
        samples, samples_idx = d_vals, None
        # Calculate Beta+ 
        deltas = d_vals[...,1:] - d_vals[...,:-1] # [..., N_init - 1]
        # NOTE: Relaxed Bound in VolSDF code base
        beta_plus = torch.sqrt((1.0 / (4.0 * torch.log(torch.tensor(self.eps + 1.0, device=device)))) * (deltas ** 2.).sum(-1))
        # NOTE: Original Bound in Paper and NeuRecon
        # beta_plus = torch.sqrt((fars**2) / (4 * (d_vals.shape[-1]-1) * np.log(1+eps)))
        
        if debug:
            print(f'beta+ from Lemma 1:',beta_plus.shape)
        total_iters, converge = 0, False
        iter_usage_map = -1 * torch.ones(*prefix,1).to(device) # iterations of convergence
        

        # Fine Sampling
        while not converge and total_iters < self.max_iter:
            # d_star calculating
            with torch.no_grad():
                if samples_idx is None:
                    pts = rays_o[..., None, :] + rays_d[..., None, :] * samples[..., :, None]
                    sdf = self.implicit_surface_fn(pts)
                else: 
                    sdf = self.query_sdf(samples, samples_idx, rays_o, rays_d, sdf) # [*prefix, 1]
            if debug:
                print('SDF:', sdf.shape)
            deltas = d_vals[...,1:] - d_vals[...,:-1] 
            delta_i, d_i_lower, d_i_upper = deltas, sdf[...,:-1], sdf[...,1:]
            first_cond = delta_i ** 2.0 + d_i_lower ** 2.0 <= d_i_upper ** 2.0
            second_cond = delta_i ** 2.0 + d_i_upper ** 2.0 <= d_i_lower ** 2.0
            d_star = torch.zeros_like(deltas, device = device)
            d_star[first_cond] = d_i_lower[first_cond]
            d_star[second_cond] = d_i_upper[second_cond]
            s = (delta_i + d_i_lower + d_i_upper)/2.0
            area_before_sqrt = s * (s - delta_i) * (s - d_i_lower) * (s - d_i_upper)
            mask = ~first_cond & ~second_cond & (d_i_lower + d_i_upper - delta_i > 0)
            d_star[mask] = (2.0 * torch.sqrt(area_before_sqrt[mask])) / (delta_i[mask])
            d_star = (sdf[..., 1:].sign() * sdf[..., :-1].sign() == 1) * d_star  # Fixing the sign

            curr_error = self.error_bound(beta0, sdf, deltas, d_star)
            beta_plus[curr_error <= self.eps] = beta0
            iter_usage_map[curr_error <= self.eps] = total_iters


            # Beta* Calculation
            beta_min, beta_max = beta0 * torch.ones_like(beta_plus,device=device), beta_plus
            for j in range(self.max_bisection):
                beta_mid = (beta_min + beta_max)/2.
                curr_error = self.error_bound(beta_mid, sdf, deltas, d_star)
                beta_max[curr_error <= self.eps] = beta_mid[curr_error <= self.eps]
                beta_min[curr_error > self.eps] = beta_mid[curr_error > self.eps]
            beta_plus=beta_max # beta+ <- beta*

            # Opacity Estimation
            sigma = self.sdf_to_sigma(sdf,1/beta_plus, beta_plus)

            deltas = torch.cat([deltas, 1e10 * torch.ones([*prefix, 1],device=device)], -1)
            left_riemann_sum = deltas*sigma
            shifted_riemann_sum = torch.cat([torch.zeros([*prefix, 1], device=device), left_riemann_sum[..., :-1]], dim=-1)
            alpha = 1-torch.exp(-left_riemann_sum)
            transmittance = torch.exp(-torch.cumsum(shifted_riemann_sum, dim=-1))
            if debug:
                print('Esitimated Transmittance:',transmittance.shape)
            weights = alpha * transmittance

            # Check Convergence
            total_iters += 1
            converge = beta_plus.max() <= beta0
            
            if debug:
                print('Converge:', converge)
                print('Iteration:', total_iters)

            # Estimate CDF
            if not converge and total_iters < self.max_iter:
                if debug:
                    print('Estimate CDF to Sample more points proportional to the current error bound')

                N = self.N_up

                bins=d_vals
                if debug:
                    print('d_star:',d_star.shape,'beta_plus',(beta_plus.unsqueeze(-1)).shape, 'deltas', deltas.shape)
                error_i =torch.exp(-d_star / beta_plus.unsqueeze(-1)) * (deltas[...,:-1] ** 2.) / (4 * beta_plus.unsqueeze(-1) ** 2)
                error_sum_i = torch.cumsum(error_i, dim=-1)
                bound_opacity = (torch.clamp(torch.exp(error_sum_i),max=1.e6) - 1.0) * transmittance[...,:-1]

                pdf = bound_opacity
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
                if debug:
                    print('Estimated CDF:', cdf.shape)

            else:
                if debug:
                    print('Estimate CDF to Sample the final sample set to be used in the volume rendering integral')

                N = self.N_fin

                bins = d_vals
                pdf = weights[..., :-1]
                pdf = pdf + 1e-5  # prevent nans
                pdf = pdf / torch.sum(pdf, -1, keepdim=True)
                cdf = torch.cumsum(pdf, -1)
                cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
                if debug:
                    print('Estimated CDF:', cdf.shape)

            # Invert CDF
            if (not converge and total_iters < self.max_iter):
                u = torch.linspace(0., 1., steps=N).to(device) * torch.ones([*prefix, N]).to(device) # [(B), N_rays, N_up]
            else:
                u = torch.rand(list(cdf.shape[:-1]) + [N],device=device)
            u = u.contiguous() 

            inds = torch.searchsorted(cdf, u, right=True) # z = cdf_inv(u) ~ cdf # [(B), N_rays, N_up]
            below = torch.max(torch.zeros_like(inds - 1), inds - 1) 
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1) # [(B), N_rays, N_up, 2]

            matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]] # [(B), N_rays, N_up, N_samples]
            cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
            bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)

            denom = (cdf_g[..., 1] - cdf_g[..., 0])
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
            
            # Adding selected samples if we not converged
            if not converge and total_iters < self.max_iter:
                d_vals, samples_idx = torch.sort(torch.cat([d_vals, samples], -1), -1)
        
        z_samples = samples
        if  debug:
            print('Final Selected Sample:', z_samples.shape) 

        if debug:
            print('Final Sample Set:', d_vals.shape)
        if self.N_extra>0:
            sampling_idx = torch.randperm(d_vals.shape[-1])[:self.N_extra]
            z_vals_extra = torch.cat([nears, fars, d_vals[...,sampling_idx]], -1)

        z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_extra], -1), -1)
        if debug:
            print('Final Coarse and Fine Samples:', z_vals.shape)

        # add some of the near surface points
        idx = torch.randint(z_vals.shape[-1], (*prefix,)).cuda()
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))
        

        return z_vals, z_samples_eik, beta_plus, iter_usage_map


