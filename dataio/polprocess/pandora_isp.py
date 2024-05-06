import sys
sys.path.append('/newdata/yokoli/neurecon')
from utils.io_util import load_rgb, glob_imgs
import numpy as np
import dataio.polanalyser as pa
import cv2
import os
import argparse
from dataio.polprocess.tools import preprocess_raw_gray, print_np, convert_u8, linear_rescale, view_dop
import imageio

parser = argparse.ArgumentParser()

parser.add_argument('--instance_dir', type=str,
                    help='input scene directory')
parser.add_argument('--img_type', type=str,
                    default='bmp',
                    help='input scene directory')
# parser.add_argument('--pose', type=str,default=False, help='convert SmartMore pose format to Colmap format')
args = parser.parse_args()

# instance_dir = './data/test/'
# img_type = 'bmp'

def cal_stokes(instance_dir, offset=3, img_type='bmp', depth = 8):
    image_dir = '{0}/sRGB'.format(instance_dir)
    image_paths = sorted(glob_imgs(image_dir))
    # image_paths = [image_path for image_path in image_paths if image_path[image_path.rfind("_")+1:image_path.rfind(".")] == "0"]
    print(image_paths)
    # image_names = [image_path[image_path.rfind("\\") + 1: image_path.rfind("_")] for image_path in image_paths]
    image_names = [os.path.basename(image_path) for image_path in image_paths]
    # image_names = [image_path[image_path.find("\\") + 1: ] for image_path in image_paths]
    print(len(image_names))
    # os.makedirs('{0}/images'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/stokes'.format(instance_dir), exist_ok=True)
    # os.makedirs('{0}/vis_stokes'.format(instance_dir), exist_ok=True)
    # os.makedirs('{0}/calibration'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/DoP_vis'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/AoP_vis'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/AoP'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/DoP'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/sRGB'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/Radiance'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/AoP_vis_sat'.format(instance_dir), exist_ok=True)
    for image_name in image_names:
        print(image_name)
        # exit()
        file_name = os.path.join(image_dir,image_name)
        # img_raw = cv2.imread(file_name, 0)
        image_name_save=image_name.split(".")[0]
        s0 = cv2.imread(f'{instance_dir}/images_stokes/{image_name_save}_s0.hdr', flags = cv2.IMREAD_UNCHANGED)
        s0p1 = cv2.imread((f'{instance_dir}/images_stokes/{image_name_save}_s0p1.hdr'), flags = cv2.IMREAD_UNCHANGED)
        s0p2 = cv2.imread((f'{instance_dir}/images_stokes/{image_name_save}_s0p2.hdr'), flags = cv2.IMREAD_UNCHANGED)
        s1 = s0p1 - s0
        s2 = s0p2 - s0
        stokes_rgb = np.stack([s0,s1,s2],-1)
        # Convert BGR to GRAY
        s0 = cv2.cvtColor(s0, cv2.COLOR_BGR2GRAY)
        s1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY)
        s2 = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY)
        stokes = np.stack([s0,s1,s2],-1)

        # Intensity Calculation
        intensity = pa.cvtStokesToIntensity(stokes_rgb)
        print(f'Max Intensity: {intensity.max()}')
        cv2.imwrite(f'{instance_dir}/Radiance/{image_name_save}.png', (255*intensity).astype('u1'))
        cv2.imwrite(f'{instance_dir}/sRGB/{image_name_save}.png', convert_u8(255*intensity))
        
        # AoP Calculation
        AoP = pa.cvtStokesToAoLP(stokes)
        np.save(f'{instance_dir}/AoP/{image_name_save}.npy', AoP)
        DoP = pa.cvtStokesToDoLP(stokes)
        DoP = np.clip(DoP,0.0, 1.0)
        print(f'Max DoP: {DoP.max()}')
        np.save(f'{instance_dir}/DoP/{image_name_save}.npy', DoP)

        # Visualization of DOP and AOP
        aop_img = pa.applyColorToAoLP(AoP)
        aop_img_sat = pa.applyColorToAoLP(AoP, saturation = DoP)
        cv2.imwrite(f'{instance_dir}/AoP_vis/{image_name_save}.png', aop_img)
        cv2.imwrite(f'{instance_dir}/AoP_vis_sat/{image_name_save}.png', aop_img_sat)

        view_dop(stokes[...,0], stokes[...,1], stokes[...,2],f'{instance_dir}/DoP_vis/{image_name_save}.png' )

if __name__ == '__main__':
    args = parser.parse_args()
    cal_stokes(args.instance_dir)