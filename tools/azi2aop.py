import numpy as np
import torch
import imageio,json, cv2
import dataio.polanalyser as pa

def normal_to_aop(pose,
                normal_map):
        '''From normals to predicted aop

        Return: 
            [B, N_rays]
        '''

        w2c = pose[:, :3,:3].transpose(1, 2) # R^T, B x 3 x 3
        # check_np(w2c)

        (N_batch, N_rays, _) = normal_map.shape

        # normal_map = normal_map.reshape([N_batch, 3, N_samples])
        normal_map = normal_map.transpose(1, 2) # [B, 3, N_rays]
        normal_map_cam = torch.bmm(w2c, normal_map) # B x 3 x 3 @ B x 3 x N_rays = B x 3 x N_rays
        normal_map_cam = normal_map_cam.transpose(1, 2) # B x N_rays x 3
        phi = torch.atan2(normal_map_cam[...,1], normal_map_cam[...,0]) # N_batch x N_rays  (rad)
        
        # MOD: PMIVR Loss Deprecated
        # eta = torch.stack([torch.abs(phi-AoP_gt-np.pi/2), torch.abs(phi-AoP_gt), torch.abs(phi-AoP_gt+np.pi/2), torch.abs(phi-AoP_gt+np.pi)], dim=1)
        # eta, _ = torch.min(eta, dim=1)

        # phi_to_aop = np.pi/2 - phi
        phi_to_aop = phi
        # mod to [0, pi]
        phi_to_aop = torch.remainder(phi_to_aop, np.pi)

        return phi_to_aop

def get_pose(cameraJson, idx):
    rotation = torch.Tensor(cameraJson[idx]['rotation']).transpose(1, 0).float() # R^T
    translation = torch.Tensor(cameraJson[idx]['camera_pos']).float() # C = -R_transpose*t
    c2w_ = torch.cat([rotation, translation.unsqueeze(1)], dim=1) # 3 x 4
    c2w = torch.cat([c2w_, torch.Tensor([[0.,0.,0.,1.]])], dim=0)
    return c2w

if __name__ == '__main__':
    normal = imageio.imread('normal.png')/255
    (H, W, _) = normal.shape
    normal_ori = (normal - 0.5)*2
    normal_ori = torch.from_numpy(normal_ori[None,...]).flatten(1,2).float()
    with open('/camera_extrinsics.json', 'r') as f:
            camera_extrinsics = json.load(f)
    c2w = get_pose(cameraJson=camera_extrinsics,idx='23.png')[None,...]  # Batched
    aop = normal_to_aop(c2w, normal_ori).numpy().reshape((H, W)).squeeze()
    print(aop.max(),aop.min())
    aop_img = pa.applyColorToAoLP(aop)
    cv2.imwrite('pred_aop_ori.png', aop_img)