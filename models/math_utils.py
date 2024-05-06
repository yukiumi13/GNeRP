import torch
import numpy as np

def _rot_matrix_from_angle(t):
    _prefix = t.shape
    rot_mat = torch.zeros(*_prefix, 2,2).type_as(t)
    rot_mat[...,0,0] = torch.cos(t)
    rot_mat[...,1,1] = torch.cos(t)
    rot_mat[...,0,1] = -torch.sin(t)
    rot_mat[...,1,0] = torch.sin(t)
    return rot_mat

def two_order_real_symmetric_matrix_svd(mat):
    """Closed-form of 2x2 symmetric real matrix svd
        [[a,b],
        [b,d]], see https://www.researchgate.net/publication/263580188_Closed_Form_SVD_Solutions_for_2_x_2_Matrices_-_Rev_2

    Args:
        mat: (...,2,2)
    """
    a, b, d = mat[...,0,0], mat[...,0,1], mat[...,1,1]
    sigmas = torch.stack([((a+d).abs() + ((a-d)**2 + 4*(b**2) + 1e-8).sqrt())/2,
                          ((a+d).abs() - ((a-d)**2 + 4*(b**2) + 1e-8).sqrt()).abs()/2], dim=-1) # (...,2)
    det_idx = a*d - b**2 < 0
    Sigma = torch.eye(2,2).type_as(sigmas).expand_as(mat).clone()
    if not det_idx.any():
        Sigma[det_idx][:,1,1] = -1
    D = torch.diag_embed(sigmas) # (...,2,2)
    Sigma *= torch.sign(a+d)[...,None,None]
    # NOTE: atan may cause nan if divisor approaching ZERO
    theta = torch.atan((-torch.sign(b)*(a-d) + torch.sign(a+d)*torch.sign(b)*((a-d)**2 + 4*b**2 + 1e-7).sqrt())/(2*b.abs() + 1e-7))
    rot_mat = _rot_matrix_from_angle(theta)
    # print(f'error:{(rot_mat @ D @ Sigma @ rot_mat.transpose(-2,-1) - mat).abs().sum()}')
    return (D @ Sigma).diagonal(dim1=-2, dim2=-1), rot_mat