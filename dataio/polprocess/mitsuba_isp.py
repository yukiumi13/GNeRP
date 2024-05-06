import sys
sys.path.append('/dataset/yokoli/neurecon')
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

def cal_stokes(instance_dir):
    n_images = len(glob_imgs(f'{instance_dir}/stokes'))//3
    print(n_images)
    # os.makedirs('{0}/images'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/stokes'.format(instance_dir), exist_ok=True)
    # os.makedirs('{0}/vis_stokes'.format(instance_dir), exist_ok=True)
    # os.makedirs('{0}/calibration'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/DoP_vis'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/AoP_vis'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/aop'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/dop'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/images'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/AoP_vis_sat'.format(instance_dir), exist_ok=True)
    for idx in range(n_images):
        print(idx)
        # exit()
        # img_raw = cv2.imread(file_name, 0)
        s0 = cv2.imread(f'{instance_dir}/stokes/{idx}_s0.hdr', flags = cv2.IMREAD_UNCHANGED)
        s0p1 = cv2.imread((f'{instance_dir}/stokes/{idx}_s0p1.hdr'), flags = cv2.IMREAD_UNCHANGED)
        s0p2 = cv2.imread((f'{instance_dir}/stokes/{idx}_s0p2.hdr'), flags = cv2.IMREAD_UNCHANGED)
        s1 = s0p1 - s0
        s2 = s0p2 - s0
        stokes_rgb = np.stack([s0,s1,s2],-1)
        # Convert BGR to GRAY
        s0 = cv2.cvtColor(s0, cv2.COLOR_BGR2GRAY)
        s1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY)
        s2 = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY)
        stokes = np.stack([s0,s1,s2],-1)

        mask = cv2.imread(f'{instance_dir}/masks/{idx}.png')

        # Intensity Calculation
        intensity = stokes_rgb[...,0]
        print(f'Max Intensity: {intensity.max()}')
        # cv2.imwrite(f'{instance_dir}/Radiance/{idx}.png', (255*intensity).astype('u1'))
        # cv2.imwrite(f'{instance_dir}/images/{idx}.png', convert_u8(255*np.clip(intensity,0,1)))
        
        # AoP Calculation
        AoP = pa.cvtStokesToAoLP(stokes)
        AoP[~mask]=0.0
        np.save(f'{instance_dir}/aop/{idx}.npy', AoP)
        DoP = pa.cvtStokesToDoLP(stokes)
        DoP[~mask]=0.0
        DoP[np.isnan(DoP)]=0.0
        DoP = np.clip(DoP,0.0, 1.0)
        print(f'Max DoP: {DoP.max()}')
        np.save(f'{instance_dir}/dop/{idx}.npy', DoP)

        # Visualization of DOP and AOP
        aop_img = pa.applyColorToAoLP(AoP)
        aop_img_sat = pa.applyColorToAoLP(AoP, saturation = DoP)
        cv2.imwrite(f'{instance_dir}/AoP_vis/{idx}.png', aop_img)
        cv2.imwrite(f'{instance_dir}/AoP_vis_sat/{idx}.png', aop_img_sat)

        view_dop(stokes[...,0], stokes[...,1], stokes[...,2],f'{instance_dir}/DoP_vis/{idx}.png' )

if __name__ == '__main__':
    args = parser.parse_args()
    cal_stokes(args.instance_dir)