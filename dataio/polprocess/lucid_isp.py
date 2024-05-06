import sys
from utils.io_util import load_rgb, glob_imgs
import numpy as np
import polanalyser_new as pa
import cv2
import os
import argparse
from polprocess.tools import preprocess_raw_gray, print_np, convert_u8, linear_rescale, view_dop
import imageio, glob

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

def calStokesFromArray(i0, i45, i90, i135, factor_0 = 1.0, factor_45 = 1.0, factor_90 = 1.0, factor_135 = 1.0):
    s0 = np.maximum((factor_0 * i0) + (factor_90 * i90), (factor_45 * i45) + (factor_135 * i135))
    s1 = (factor_0 * i0) - (factor_90 * i90)
    s2 = (factor_45 * i45) - (factor_135 * i135)
    return np.stack([s0, s1, s2], -1)

def lucid_isp(instance_dir, img_type='jpg', depth = 8):
    polarizerDir = f'{instance_dir}/Polarizer'
    imgPath = sorted(glob.glob(f'{polarizerDir}/*90deg.{img_type}'))
    print(imgPath)
    imgList=[]
    for p in imgPath:
        imgName = os.path.basename(p).split('.')[-2].split('_')[:-1]
        imgName = '_'.join(imgName)
        imgList.append(imgName)

    os.makedirs('{0}/DoP_vis'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/AoP_vis'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/AoP'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/DoP'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/sRGB'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/Radiance'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/AoP_vis_sat'.format(instance_dir), exist_ok=True)
    os.makedirs('{0}/Stokes'.format(instance_dir), exist_ok=True)
    
    print(f'{len(imgList)} Images: \n', imgList)
    
    for imgName in imgList:
        img0 = cv2.imread(f'{polarizerDir}/{imgName}_0.{img_type}')[:,:,0]
        img45 = cv2.imread(f'{polarizerDir}/{imgName}_45.{img_type}')[:,:,0]
        img90 = cv2.imread(f'{polarizerDir}/{imgName}_90.{img_type}')[:,:,0]
        img135 = cv2.imread(f'{polarizerDir}/{imgName}_135.{img_type}')[:,:,0]
        (H, W) = img0.shape
        

        # CFA -> CPFA
        imgRaw = np.zeros((2*H, 2*W))
        imgRaw[::2, ::2] = img90/255.
        imgRaw[1::2, ::2] = img135/255.
        imgRaw[::2, 1::2] = img45/255.
        imgRaw[1::2, 1::2] = img0/255.

        img0, img45, img90, img135 = pa.demosaicing(imgRaw, pa.COLOR_PolarRGB)        
        img_list = [img0, img45, img90, img135]
        img_list = np.stack(img_list,-1)
        angles = np.deg2rad([0, 45, 90, 135])
        # img_stokes = calStokesFromArray(img0, img45, img90, img135)
        three_pinv = 1
        if not three_pinv:
            img_stokes = (pa.calcStokes(img_list, angles))
        else:
        # Select angle that has highest intensity and remove it 
            max_angle_ind = np.argmax(img_list.sum(axis=(0,1,2)))
            img_pp_channel_rem = [img_list[:,:,:,a] for a in range(len(angles)) if a != max_angle_ind]
            angles_rem = [angles[a] for a in range(len(angles)) if a!= max_angle_ind]
            img_stokes = pa.calcStokes(img_pp_channel_rem, angles_rem)
        # img_stokes = pa.calcStokes(img_list, angles).astype('f4')
        s0, s1, s2 = (img_stokes[:,:,:,0]/2.0).astype('f4'), (img_stokes[:,:,:,1]/2.0).astype('f4'), (img_stokes[:,:,:,2]/2.0).astype('f4')
        stokes_rgb = np.stack([s0,s1,s2],-1)
        # Convert BGR to GRAY
        s0 = cv2.cvtColor(s0, cv2.COLOR_BGR2GRAY)
        s1 = cv2.cvtColor(s1, cv2.COLOR_BGR2GRAY)
        s2 = cv2.cvtColor(s2, cv2.COLOR_BGR2GRAY)
        stokes = np.stack([s0,s1,s2],-1)

        # Intensity Calculation
        intensity = pa.cvtStokesToIntensity(stokes_rgb)
        print(f'Max Intensity: {intensity.max()}')
        cv2.imwrite(f'{instance_dir}/Radiance/{imgName}.png', (255*intensity).astype('u1'))
        cv2.imwrite(f'{instance_dir}/sRGB/{imgName}.png', convert_u8(255*intensity))
        
        # AoP Calculation
        AoP = pa.cvtStokesToAoLP(stokes)
        np.save(f'{instance_dir}/AoP/{imgName}.npy', AoP)
        DoP = pa.cvtStokesToDoLP(stokes)
        idx = np.isnan(DoP)
        DoP[idx]=0.0
        DoP = np.clip(DoP,0.0, 1.0)
        print(f'Max DoP: {DoP.max()}')
        np.save(f'{instance_dir}/DoP/{imgName}.npy', DoP)

        # Visualization of DOP and AOP
        aop_img = pa.applyColorToAoLP(AoP)
        aop_img_sat = pa.applyColorToAoLP(AoP, saturation = DoP)
        cv2.imwrite(f'{instance_dir}/AoP_vis/{imgName}.png', aop_img)
        cv2.imwrite(f'{instance_dir}/AoP_vis_sat/{imgName}.png', aop_img_sat)

        view_dop(stokes[...,0], stokes[...,1], stokes[...,2],f'{instance_dir}/DoP_vis/{imgName}.png' )

if __name__ == '__main__':
    args = parser.parse_args()
    # cal_stokes(args.instance_dir)
    lucid_isp('pol/Hulk', img_type = 'png')