import os
import sys
sys.path.append('/dataset/yokoli/neurecon')

import json
import numpy as np
import argparse

def npz2json(root_path):
    # root_path = '/newdata/yokoli/neurecon/data/pol/o'
    camera_path = os.path.join(root_path, 'cameras.npz')

    save_path = os.path.join(root_path, 'cameras.json')

    cameras = {}

# import IPython; IPython.embed(); exit()

# for i in range(0, len(lines), 14):
#     filename = lines[i+1][:-1]

#     cameras[filename] = dict(
#         focal_length = list(map(float,lines[i+3][:-1].split(' '))),
#         image_center = list(map(float,lines[i+4][:-1].split(' '))),
#         translation = list(map(float,lines[i+5][:-2].split(' '))),
#         camera_pos = list(map(float,lines[i+6][:-2].split(' '))),
#         quaternion = list(map(float,lines[i+8][:-2].split(' '))),
#         rotation = [list(map(float, lines[i+9][:-2].split(' '))), 
#                 list(map(float, lines[i+10][:-2].split(' '))), 
#                 list(map(float, lines[i+11][:-2].split(' ')))],
#     )


    camera_dict = np.load(camera_path)
    image_names = sorted([k[10:] for k in camera_dict.keys() if k.startswith('scale_mat_')])
    N_images = len(image_names)
    print(f'{N_images} Detected.')
    scale_mats = [camera_dict[f'scale_mat_{image_name}'].astype(np.float32) for image_name in image_names]
    world_mats = [camera_dict[f'world_mat_{image_name}'].astype(np.float32) for image_name in image_names]

    for img_name, scale_mat, world_mat in zip(image_names, scale_mats, world_mats):
        print(f'Processing {img_name}')
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(P)
        translation = -pose[:3,:3].T @ pose[:3,3]
        cameras[f'{img_name}.png'] = dict(
        focal_length = intrinsics[[0,1],[0,1]].astype(np.float32).tolist(),
        image_center = intrinsics[[0,1],[2,2]].astype(np.float32).tolist(),
        translation = translation.astype(np.float32).tolist(),
        rotation = pose[:3,:3].T.astype(np.float32).tolist(), # c2w -> w2c
        camera_pos=pose[:3,3].astype(np.float32).tolist()
        )

    json_str = json.dumps(cameras, indent=4)
    with open(save_path, 'w') as json_file:
        json_file.write(json_str)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./')
    opt = parser.parse_args()
    npz2json(opt.path)

