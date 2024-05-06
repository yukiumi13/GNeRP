import cv2
import numpy as np
import torch


def applyColorToAoLP(img_AoLP, saturation=1.0, value=1.0):
    """
    Apply color map to AoLP image
    The color map is based on HSV
    Parameters
    ----------
    img_AoLP : np.ndarray, (height, width)
        AoLP image. The range is from 0.0 to pi.
    
    saturation : float or np.ndarray, (height, width)
        Saturation part (optional).
        If you pass DoLP image (img_DoLP) as an argument, you can modulate it by DoLP.
    value : float or np.ndarray, (height, width)
        Value parr (optional).
        If you pass DoLP image (img_DoLP) as an argument, you can modulate it by DoLP.
    """
    img_ones = np.ones_like(img_AoLP)

    img_hue = (np.mod(img_AoLP, np.pi)/np.pi*179).astype(np.uint8) # 0~pi -> 0~179
    img_saturation = np.clip(img_ones*saturation*255, 0, 255).astype(np.uint8)
    img_value = np.clip(img_ones*value*255, 0, 255).astype(np.uint8)
    
    img_hsv = cv2.merge([img_hue, img_saturation, img_value])
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def batchApplyColorToAoLP(img_AoLP, saturation=1.0, value=1.0):
    """
    Apply color map to AoLP image
    The color map is based on HSV
    Parameters
    ----------
    img_AoLP : np.ndarray, (B, height, width)
        AoLP image. The range is from 0.0 to pi.
    
    saturation : float or np.ndarray, (height, width)
        Saturation part (optional).
        If you pass DoLP image (img_DoLP) as an argument, you can modulate it by DoLP.
    value : float or np.ndarray, (height, width)
        Value parr (optional).
        If you pass DoLP image (img_DoLP) as an argument, you can modulate it by DoLP.
    """
    imgs_rgb = np.zeros((*img_AoLP.shape,3))
    for i in range(img_AoLP.shape[0]):
        if type(saturation) is float:
            imgs_rgb[i,...] = applyColorToAoLP(img_AoLP[i,...],saturation=saturation)
        else:
            imgs_rgb[i,...] = applyColorToAoLP(img_AoLP[i,...],saturation=saturation[i,...])

    return imgs_rgb


def normal_to_aop(normal_map_cam, opengl = False):
        '''From normals to predicted aop
        Args:
            [N_rays, 3]
        Return: 
            [N_rays]
        '''

        phi = torch.atan2(normal_map_cam[...,1], normal_map_cam[...,0]) # N_batch x N_rays  (rad) 
        
        phi_to_aop = np.pi/2 + phi if opengl else np.pi/2 - phi
        phi_to_aop = torch.remainder(phi_to_aop, np.pi)

        return phi_to_aop

def world_normal_to_aop(pose,
                normal_map):
        '''From normals to predicted aop

        Return: 
            [B, N_rays]
        '''

        w2c = pose[:, :3,:3].transpose(1, 2) # R^T, B x 3 x 3
        # check_np(w2c)

        (N_batch, N_rays, _) = normal_map.shape

        # N_samples = normal_map.shape[0] // batch_size

        # print('get_AoP_loss', 'N_samples', N_samples)
        # print('get_AoP_loss', 'Batch_size', batch_size)

        # B x S --> S ONLY work if Batch Size is 1
        # AoP_gt = AoP_gt[0]
        # DoP_gt = DoP_gt[0]
        # mask = mask[0]

        # print('DoP_gt', DoP_gt.shape)
        # print('AoP_gt', AoP_gt.shape)


        # normal_map = normal_map.reshape([N_batch, 3, N_samples])
        normal_map = normal_map.transpose(1, 2) # [B, 3, N_rays]
        normal_map_cam = torch.bmm(w2c, normal_map) # B x 3 x 3 @ B x 3 x N_rays = B x 3 x N_rays
        normal_map_cam = normal_map_cam.transpose(1, 2) # B x N_rays x 3
        phi = torch.atan2(normal_map_cam[...,1], normal_map_cam[...,0]) # N_batch x N_rays  (rad)
        
        # MOD: PMIVR Loss Deprecated
        # eta = torch.stack([torch.abs(phi-AoP_gt-np.pi/2), torch.abs(phi-AoP_gt), torch.abs(phi-AoP_gt+np.pi/2), torch.abs(phi-AoP_gt+np.pi)], dim=1)
        # eta, _ = torch.min(eta, dim=1)

        phi_to_aop = np.pi/2 + phi
        # mod to [0, pi]
        phi_to_aop = torch.remainder(phi_to_aop, np.pi)

        return phi_to_aop

def normal_to_dop(pose, normal_map):

        w2c = pose[:, :3,:3].transpose(1, 2) # R^T, B x 3 x 3
        # check_np(w2c)

        (N_batch, N_rays, _) = normal_map.shape

        # N_samples = normal_map.shape[0] // batch_size

        # print('get_AoP_loss', 'N_samples', N_samples)
        # print('get_AoP_loss', 'Batch_size', batch_size)

        # B x S --> S ONLY work if Batch Size is 1
        # AoP_gt = AoP_gt[0]
        # DoP_gt = DoP_gt[0]
        # mask = mask[0]

        # print('DoP_gt', DoP_gt.shape)
        # print('AoP_gt', AoP_gt.shape)


        # normal_map = normal_map.reshape([N_batch, 3, N_samples])
        normal_map = normal_map.transpose(1, 2) # [B, 3, N_rays]
        normal_map_cam = torch.bmm(w2c, normal_map) # B x 3 x 3 @ B x 3 x N_rays = B x 3 x N_rays
        normal_map_cam = normal_map_cam.transpose(1, 2) # B x N_rays x 3
        image_plane_norm = normal_map_cam[...,:-1].norm(2,-1) # [B, N_rays]
        zenith_angle = torch.atan2(image_plane_norm, normal_map_cam[...,2])
        # DEBUG
        # print('zenith_angle', zenith_angle.min(), zenith_angle.max())
        n_ref = 1.5 * torch.ones_like(zenith_angle) # Refraction Index Default 1.5
        s2 = torch.sin(zenith_angle).pow(2)
        s4 = s2.pow(2)
        c2 = torch.cos(zenith_angle).pow(2)
        dop = torch.sqrt(s4*c2*(n_ref.pow(2)-s2))/((s4 +c2*(n_ref.pow(2) - s2))/2)

        return dop


def check_np(x:np.ndarray):
    print(x.shape)
    print(f'[{x.min():02f},{x.max():02f}]')

def visualize_aop(aop, dop=1.0):
        '''AoP -> HSV -> RGB

        Args:
            aop, dop (N,)
        '''
        ones = torch.ones_like(aop)

        hue = (torch.remainder(aop, np.pi)/np.pi*179).type(torch.uint8).cpu().numpy() # 0~pi -> 0~179
        saturation = torch.clamp(ones*dop*255, 0, 255).type(torch.uint8).cpu().numpy()
        value = torch.clamp(ones*255, 0, 255).type(torch.uint8).cpu().numpy()
    
        img_hsv = cv2.merge([hue, saturation, value])
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return torch.from_numpy(img_rgb)/255.