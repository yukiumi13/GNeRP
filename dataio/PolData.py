import sys
sys.path.append('/neurecon')
import os
import torch
import numpy as np

import json

from utils import rend_util
from tqdm import tqdm
from utils.io_util import load_mask, load_rgb, load_exr, glob_imgs, load_npy, load_gray, load_mask_u8
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
from functools import partial
from utils.log_utils import pretty_table_log

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 train_cameras,
                 data_dir,   
                 downscale = 1,
                 scale_radius = -1,
                 chromatic = 'Mono',
                 opengl = False,
                 crop_quantile = None,
                 ):

        self.instance_dir = data_dir
        self.scale_radius = scale_radius
        assert os.path.exists(self.instance_dir), "Data directory is empty"
        self.opengl = opengl
        self.has_normals = os.path.exists(os.path.join(self.instance_dir, 'normals'))

        self.train_cameras = train_cameras
        if chromatic == 'Mono':
            image_dir = '{0}/images'.format(self.instance_dir)
        elif chromatic == 'sRGB':
            image_dir = '{0}/images'.format(self.instance_dir)
        # image_paths = sorted(glob_imgs(image_dir))
        mask_dir = '{0}/masks'.format(self.instance_dir)
        # mask_paths = sorted(glob_imgs(mask_dir))
        mask_ignore_dir = '{0}/mask_ignore'.format(self.instance_dir)

        self.has_mask = os.path.exists(mask_dir) and len(os.listdir(mask_dir)) > 0
        self.has_mask_out = os.path.exists(mask_ignore_dir) and len(os.listdir(mask_ignore_dir)) > 0

        image_paths = sorted(glob_imgs(image_dir))
        self.n_images = len(image_paths)

        with open(os.path.join(self.instance_dir, 'cameras.json'), 'r') as f:
            camera_intrinsics = json.load(f)['intrinsic']

        with open(os.path.join(self.instance_dir, 'cameras.json'), 'r') as f:
            camera_extrinsics = json.load(f)
        
        # self.n_images = len(camera_extrinsics) 
        camera_intrinsics = np.array(camera_intrinsics,dtype=np.float32).reshape(3,3)

        # import IPython; IPython.embed(); exit()
        cam_center_norms = []
        self.intrinsics_all = []
        self.c2w_all = []
        self.rgb_images = []
        self.object_masks = []
        self.masks_ignore = []
        
        if self.has_normals:
            self.normals = []

        # Pol Dataset
        self.AoP_maps = []
        self.DoP_maps = []

        # downscale intrinsics
        camera_intrinsics[0, 2] /= downscale
        camera_intrinsics[1, 2] /= downscale
        camera_intrinsics[0, 0] /= downscale
        camera_intrinsics[1, 1] /= downscale

        if crop_quantile is not None:
            camera_intrinsics[0, 2] /= float(crop_quantile[0])
            camera_intrinsics[1, 2] /= float(crop_quantile[1])
        

        for idx in tqdm(range(self.n_images), desc="Loading data..."):
        # for idx in range(self.n_images):
            file_name = f'{idx}'
            file_name_png = f'{idx}.png'
            file_name_npy =  f'{idx}.npy'
            self.intrinsics_all.append(torch.Tensor(camera_intrinsics).float())

            # Pose (c2w)
            w2c_mat = torch.Tensor(camera_extrinsics[f'w2c_mat_{idx}']).float()
            rotation = w2c_mat[:3,:3].T
            translation = -rotation @ w2c_mat[:3,3:] 
            cam_center_norms.append(np.linalg.norm(translation.numpy()))
            c2w_ = torch.cat([rotation, translation], dim=1) # 3 x 4
            c2w = torch.cat([c2w_, torch.Tensor([[0.,0.,0.,1.]])], dim=0)
            if self.opengl:
                c2w[:3,1:3] *= -1.
            self.c2w_all.append(c2w) # 4 x 4

            # Load RGB images
            if chromatic == 'sRGB':
            # RGB
                rgb = load_rgb(os.path.join(image_dir, file_name_png), downscale)
                _, self.H, self.W = rgb.shape
            elif chromatic == 'Mono':
                # Mono
                rgb = load_gray(os.path.join(image_dir, file_name_png), downscale)
                self.H, self.W = rgb.shape
                rgb = rgb[None,...].repeat(3,axis=0)
            rgb = torch.from_numpy(rgb).float()
            if crop_quantile is not None:
                self.H, self.W = self.H // crop_quantile[1], self.W // crop_quantile[0]
                crop_handle = partial(TF.center_crop, output_size = [self.H, self.W])
            
            # Load Masks
            if self.has_mask:
                object_mask = load_mask_u8(os.path.join(mask_dir, "{}.png".format(file_name)), downscale)
                object_mask = torch.from_numpy(object_mask).to(dtype=torch.bool)
            if self.has_mask_out:
                mask_ignore = load_mask_u8(os.path.join(mask_ignore_dir, "{}.png".format(file_name)), downscale)
                mask_ignore = torch.from_numpy(mask_ignore).to(dtype=torch.bool)
                if crop_quantile is not None:
                    mask_ignore = crop_handle(mask_ignore)
                self.masks_ignore.append(mask_ignore.reshape(-1))

            if self.has_normals:
                gt_normal_map = load_exr(os.path.join(self.instance_dir, 'normals', f'{idx}.exr'))
                self.normals.append(F.normalize(torch.from_numpy(gt_normal_map),dim=2))

            # Load Pol Data
            AoP_map = load_npy(os.path.join(self.instance_dir, 'aop', file_name_npy), downscale = downscale)
            DoP_map = load_npy(os.path.join(self.instance_dir, 'dop',file_name_npy), downscale = downscale)
            AoP_map = torch.from_numpy(AoP_map).float()
            DoP_map = torch.from_numpy(DoP_map).float()
            # Crop
            if crop_quantile is not None:
                rgb, object_mask, AoP_map, DoP_map = list(map(crop_handle, [rgb, object_mask, AoP_map, DoP_map]))

            # Flatten Shape    
            self.rgb_images.append(rgb.reshape(3,-1).T)
            self.object_masks.append(object_mask.reshape(-1))
            self.AoP_maps.append(AoP_map.reshape(-1))
            self.DoP_maps.append(DoP_map.reshape(-1))
        
        
        max_cam_norm = max(cam_center_norms)
        min_cam_norm = min(cam_center_norms)

        camNormScaled = []
        if scale_radius > 0:
            for i in range(len(self.c2w_all)):
                self.c2w_all[i][:3, 3] *= (scale_radius / max_cam_norm)
                camNormScaled.append(np.linalg.norm(self.c2w_all[i][:3,3].numpy()))
            # import IPython; IPython.embed(); exit()
        config_field = ['scene','chromatic', 'HW', 'mask','normals_available','n_images', 'OpenGL', 'camera_radius', 'scaled_camera_radius']
        values = [self.instance_dir.split('/')[-1],chromatic,f'{self.H}x{self.W}',self.has_mask,self.has_normals,self.n_images, self.opengl, f'{min_cam_norm:.3f} ~ {max_cam_norm:.3f}', 
                  f'{min(camNormScaled):.3f}~{max(camNormScaled):.3f}']
        pretty_table_log(config_field, values)
    # import IPython; IPython.embed(); exit()

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        # uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        # uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        # uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            # "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            # "pose": self.pose_all[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx],
            "AoP_map": self.AoP_maps[idx],
            "DoP_map": self.DoP_maps[idx]
        }
        if self.has_mask:

            ground_truth["mask"]=self.object_masks[idx]

        if self.has_mask_out:

            ground_truth["mask_ignore"] = self.masks_ignore[idx]

        # if self.sampling_idx is not None:
        #     # ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]

        #     ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx]
        #     sample["uv"] = uv[self.sampling_idx, :]

        #     ground_truth["AoP_map"] = self.AoP_maps[idx][self.sampling_idx]
        #     ground_truth["DoP_map"] = self.DoP_maps[idx][self.sampling_idx]
        #     ground_truth['mask'] = self.object_masks[idx][self.sampling_idx]

        if not self.train_cameras:
            sample["c2w"] = self.c2w_all[idx]
        
        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def get_gt_pose(self, scaled=True):
        # Load gt pose without normalization to unit sphere
        with open(os.path.join(self.instance_dir, 'cameras.json'), 'r') as f:
            camera_extrinsics = json.load(f)

        c2w_all = []
        for idx in range(len(camera_extrinsics)-1):
            w2c_mat = torch.Tensor(camera_extrinsics[f'w2c_mat_{idx}']).float()
            rotation = w2c_mat[:3,:3].T
            translation = -rotation @ w2c_mat[:3,3:] 
            if scaled:
                translation /= self.scale_radius
            c2w_ = torch.cat([rotation, translation], dim=1) # 3 x 4
            c2w = torch.cat([c2w_, torch.Tensor([[0.,0.,0.,1.]])], dim=0)
            c2w_all.append(c2w) # 4 x 4

        return torch.cat([p.float().unsqueeze(0) for p in c2w_all], 0)

    # def change_sampling_idx(self, sampling_size):
    #     if sampling_size == -1:
    #         self.sampling_idx = None
    #     else:
    #         self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    # def get_scale_mat(self):
    #     assert -1, "Should not run here!"
    #     # return np.load(self.cam_file)['scale_mat_0']

if __name__ == "__main__":
    dataset = SceneDataset(False, '/dataset/pol/pmivr_car', chromatic='sRGB', scale_radius=3,opengl=True)
    c2w = dataset.get_gt_pose(scaled=True).data.cpu().numpy()
    extrinsics = np.linalg.inv(c2w)  # camera extrinsics are w2c matrix
    camera_matrix = next(iter(dataset))[1]['intrinsics'].data.cpu().numpy()

    from tools.vis_camera import visualize
    visualize(camera_matrix, extrinsics)

