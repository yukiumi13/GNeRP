import numpy as np
import torch
import plotly.offline as offline
import plotly.graph_objs as go
import torch.nn.functional as F

def perspective_opengl(cx, cy, n, f):
    """Camera Space to Orthogonal Space

    Args:
        cx: (r-l)/2
        cy: (t-b)/2
        n: near >0
        f: far >0
    """
    return torch.Tensor([[n/cx,0,0,0],
                         [0,n/cy,0,0],
                         [0,0,-(f+n)/(f-n), -2*n*f/(f-n)],
                         [0,0,-1,0]]).type_as(n)

def _batch_linear_transform(A, x):
    """batched matrix multiplication

    Args:
        A: [m,n]
        x: [N,n]
    """ 
    return (A@(x[...,None])).squeeze(-1)

def _cart_to_homo(x, w=1.0):
    return torch.cat([x, w*torch.ones_like(x[...,:1])],-1)

def _homo_to_cart(x):
    return (x/x[...,3:])[...,:3]

def perspective_to_orthogonal(pts, near, far):
    """Perspective to orthogonal

    Args:
        pts: N,3 
        near: >0
        far: >0
    """    
    trans_matrix = torch.Tensor([[near,0,0,0],[0, near, 0, 0],[0,0,near+far, near*far],[0,0,-1,0]]).type_as(pts)
    return _homo_to_cart(_batch_linear_transform(trans_matrix, _cart_to_homo(pts)))

def perspective_to_orthogonal_homo_coord(pts, near, far):
    """Perspective to orthogonal

    Args:
        pts: N,3 
        near: >0
        far: >0
    """    
    trans_matrix = torch.Tensor([[near,0,0,0],[0, near, 0, 0],[0,0,near+far, near*far],[0,0,-1,0]]).type_as(pts)
    return (_batch_linear_transform(trans_matrix, _cart_to_homo(pts)))

def orthogonal_to_perspective(pts, near, far):
    """Perspective to orthogonal

    Args:
        pts: N,3 
        near: >0
        far: >0
    """    
    #NOTE: torch.inverse() doesn't support multi-threading, see https://github.com/pytorch/pytorch/issues/90613, use hard code
    trans_matrix = torch.Tensor([[1/near,0,0,0],[0, 1/near, 0, 0],[0,0,0, -1],[0,0,1/(near*far),(near+far)/(near*far)]]).type_as(pts)
    return _homo_to_cart(_batch_linear_transform(trans_matrix, _cart_to_homo(pts)))

def orthogonal_to_perpective_homo_coord(pts, near, far):
    """Perspective to orthogonal

    Args:
        pts: N,4 
        near: >0
        far: >0
    """    
    trans_matrix = torch.Tensor([[near,0,0,0],[0, near, 0, 0],[0,0,near+far, near*far],[0,0,-1,0]]).type_as(pts).inverse()
    return _homo_to_cart((_batch_linear_transform(trans_matrix, pts)))

def perspective_to_ray_space(pts):
    """Used in https://www.cs.umd.edu/~zwicker/publications/EWAVolumeSplatting-VIS01.pdf

    Args:
        pts: [N, 3]

    Returns:
        pts
    """    
    z = pts.norm(p=2,dim=-1,keepdim=False)
    pts[:,0] /= pts[:,2]
    pts[:,1] /= pts[:,2]
    pts[:,2] = z
    return pts

def rays_uni_sample(rays_o, rays_d, near = None, far = None, samples = 10):
    if far is not None:
        t_ = far * torch.ones_like(rays_o[...,0])
    else:
        t_= 10. * torch.ones_like(rays_o[...,0])
    if near is not None:
        rays_o = rays_o  + near * rays_d
    k = torch.linspace(0,1,samples).type_as(rays_o)
    rays_t = (rays_o[None,...] + k[...,None,None] * t_[None,...,None] * rays_d[None,...]).reshape(-1,3)

    return rays_t

def world_to_camera(c2w, pts):
    R_t = c2w[:3,:3].T
    t = - R_t @ c2w[:3,3]
    return _batch_linear_transform(R_t, pts) + t

def world_to_camera_orient(c2w, pts):
    R_t = c2w[:3,:3].T
    return _batch_linear_transform(R_t, pts)

def camera_projection( 
                    normal_map,
                    pose):
        w2c = pose[:, :3,:3].transpose(1, 2) # R^T, B x 3 x 3
        (N_batch, N_rays,_) = normal_map.shape
        normal_map = normal_map.transpose(1, 2) # [B, 3, N_rays]
        normal_map_cam = torch.bmm(w2c, normal_map) # B x 3 x 3 @ B x 3 x N_rays = B x 3 x N_rays
        normal_map_cam = normal_map_cam.transpose(1, 2) # B x N_rays x 3
        return normal_map_cam

def plot_pts(pts, filename='points'):
    _pts = pts.cpu().numpy()
    pts = go.Scatter3d(x=_pts[:,0],
                        y=_pts[:,1],
                        z=_pts[:,2],
                        mode='markers',
                        marker=dict(size=1))
    fig = go.Figure(data=[pts])
    offline.plot(fig, filename=f'{filename}.html', auto_open=False)

def plot_pts_with_neighborhood(pts, pts_2, filename='points'):
    _pts = pts.cpu().numpy()
    _pts_2 = pts_2.cpu().numpy()
    pts = go.Scatter3d(x=_pts[:,0],
                        y=_pts[:,1],
                        z=_pts[:,2],
                        mode='markers',
                        marker=dict(size=1))
    pts_2 = go.Scatter3d(x=_pts_2[:,0],
                        y=_pts_2[:,1],
                        z=_pts_2[:,2],
                        mode='markers',
                        marker=dict(size=1, color='yellow'))
    fig = go.Figure(data=[pts, pts_2])
    offline.plot(fig, filename=f'{filename}.html', auto_open=False)

def plot_normals(normals, pts, filename= 'normals', interval = None):
    _pts, _normals = pts.cpu().numpy(), F.normalize(normals).cpu().numpy()
    if interval is not None:
        _pts, _normals = _pts[::interval], _normals[::interval]
    xyz_range = _pts.max() - _pts.min()
    step_size = xyz_range / 50
    pts_end = _pts + step_size * _normals
    _pts = np.stack([_pts, pts_end], axis = -1)
    lines = []
    colors = ((_normals + 1) / 2 * 255).astype('u1')
    for i in range(_pts.shape[0]):
        lines.append(go.Scatter3d(x = [_pts[i,0,0],_pts[i,0,1]],
                         y = [_pts[i,1,0],_pts[i,1,1]],
                         z = [_pts[i,2,0],_pts[i,2,1]],
                         mode = 'lines',
                         # marker = dict(showscale=False),
                         line = dict(color='rgb({},{},{})'.format(*colors[i]),
                                     width=5))
                    )                 
    dists = _pts[:,:,1] - _pts[:,:,0]
    cones = go.Cone(x=_pts[:,0,1],
                        y=_pts[:,1,1],
                        z=_pts[:,2,1],
                        u = 0.3 * dists[:,0],
                        v = 0.3 * dists[:,1],
                        w = 0.3 * dists[:,2],
                        sizeref=0.5,
                        name='cones',
                        showscale=False)
    
    fig = go.Figure(data=lines + [cones])
    fig.update_layout(showlegend=False)
    offline.plot(fig, filename=f'{filename}.html', auto_open=False)

def plot_vector_field(normals, pts, filename= 'vector_field', interval = None):
    _pts, _normals = pts.cpu().numpy(), F.normalize(normals).cpu().numpy()
    if interval is not None:
        _pts, _normals = _pts[::interval], _normals[::interval]
    xyz_range = _pts.max() - _pts.min()
    step_size = xyz_range / 50
    pts_end = _pts + step_size * _normals
    _pts = np.stack([_pts, pts_end], axis = -1)
    lines = []
    colors = ((_normals + 1) / 2 * 255).astype('u1')
    for i in range(_pts.shape[0]):
        lines.append(go.Scatter3d(x = [_pts[i,0,0],_pts[i,0,1]],
                         y = [_pts[i,1,0],_pts[i,1,1]],
                         z = [_pts[i,2,0],_pts[i,2,1]],
                         mode = 'lines',
                         # marker = dict(showscale=False),
                         line = dict(color='rgb({},{},{})'.format(*colors[i]),
                                     width=5))
                    )                 
    dists = _pts[:,:,1] - _pts[:,:,0]
    cones = go.Cone(x=_pts[:,0,1],
                        y=_pts[:,1,1],
                        z=_pts[:,2,1],
                        u = 0.3 * dists[:,0],
                        v = 0.3 * dists[:,1],
                        w = 0.3 * dists[:,2],
                        sizeref=0.5,
                        name='cones',
                        showscale=False)
    points_plot = go.Scatter3d(x = _pts[:,0,0],
                               y = _pts[:,1,0],
                               z = _pts[:,2,0],
                               mode = 'markers',
                               marker = dict(color = _pts[:,2,0], size = 3))
    fig = go.Figure(data=lines + [cones] + [points_plot])
    fig.update_layout(showlegend=False)
    offline.plot(fig, filename=f'{filename}.html', auto_open=False)

def create_step_vectors(pts):
    ones_vec = torch.ones([3,4]).type_as(pts)
    ones_vec[0,2:] = -1
    ones_vec[1,1::2] = -1
    ones_vec[2,:] = 0
    return [ones_vec[:,i] for i in range(4)]

def create_step_vectors_2x(pts):
    ones_vec = torch.ones([3,8]).type_as(pts)
    ones_vec[[0,0],[2,3]] = -1
    ones_vec[[1,1],[1,3]] = -1
    ones_vec[:,4:] = 0.5*ones_vec[:,:4]
    ones_vec[2,:] = 0
    return [ones_vec[:,i] for i in range(8)]

def create_step_vectors_4x(pts):
    ones_vec = torch.ones([3,16]).type_as(pts)
    ones_vec[[0,0],[2,3]] = -1
    ones_vec[[1,1],[1,3]] = -1
    ones_vec[0,6:] = 0
    ones_vec[0,5] = -1
    ones_vec[1,4:6] = 0
    ones_vec[1,7] = -1
    ones_vec[2,:] = 0
    return [ones_vec[:,i] for i in range(8)]

def indexing_2d_samples(select_inds, H, W, scale_pixel = 1):
    _bound = H * W - 1
    _up = select_inds - scale_pixel * W
    _up = torch.where(_up > 0, _up, select_inds)
    _down = select_inds + scale_pixel * W
    _down = torch.where(_down < _bound, _down, select_inds)
    _right = select_inds + scale_pixel
    _right = torch.where(_right < _bound, _right, select_inds)
    _left = select_inds - scale_pixel
    _left = torch.where(_left > 0, _left, select_inds)
    return torch.stack([_up, _down, _left, _right], -1) # [1, N_rays, 4]

def congruent_transform(sigma, V):
    V = V.expand_as(sigma)
    return V @ sigma @ V.transpose(-1,-2)