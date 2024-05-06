from re import S
import numpy as np 
import cv2
from .. import polanalyser as pa
from skimage.transform import rescale
import glob
import imageio

def print_np (x,show_print=False, show_shape =False, show_dtype=False, show_range=True):
    if show_print:
        print(x)
    if show_shape:
        print(x.shape)
    if show_dtype:
        print(x.dtype)
    if show_range:
        print('[', x.min(), ',', x.max(), ']')

def linear_rescale(x,min =0, max =255):
    y = ((x - 0)/(x.max() - 0))*(max - min) + min
    return y

def convert_u8(image, gamma=1/2.2):
    '''Gamma Correction'''
    image = np.clip(image, 0, 255).astype(np.uint8)
    lut = (255.0 * (np.linspace(0, 1, 256) ** gamma)).astype(np.uint8)
    return lut[image]

def preprocess_raw_gray(img_raw, scale=1., thres=1.,depth=8):
    # import time
    # t = time.time()
    out = {}
    img_raw = img_raw.astype('float32')/(2**(depth)-1)
    img_raw = scale*img_raw # H x W float
    img_raw = np.minimum(img_raw, thres)
    img_pp = pa.demosaicing(img_raw)

    # import imageio; imageio.imwrite('viz/img_s0.png',img_demosaiced[...,0].astype('float32')/4096)


    angles = np.deg2rad([0, 45, 90, 135])
    # elapsed=time.time() - t
    # print(f'Preprocessing time {elapsed}')
    # t = time.time()
    three_pinv = 1
    if not three_pinv:
        img_stokes_channel = (pa.calcStokes(img_pp, angles))
    else:
        # Select angle that has highest intensity and remove it 
        max_angle_ind = np.argmax(img_pp.sum(axis=(0,1,2)))
        img_pp_channel_rem = np.stack([img_pp[:,:,a] 
                                                for a in range(len(angles)) 
                                            if a != max_angle_ind],-1)
        angles_rem = [angles[a] for a in range(len(angles)) 
                                if a!= max_angle_ind]
        img_stokes_channel = pa.calcStokes(img_pp_channel_rem, angles_rem)
        
    # elapsed=time.time() - t
    # print(f'Initial separation time {elapsed}')
    out['stokes'] = img_stokes_channel #HxWx3(Stokes)

    return out

def demosaic_color_and_upsample(img_raw):
    #img_raw: CPFA H x W x 3 
    # may should be H x W ?
    #return: img_pfa_rgb: H x W x 3 x 4 
    # may should be H//2 x W//2 x 3 x 4 and no upsampling
    height, width = img_raw.shape[:2]
    img_pfa_rgb = np.empty((height//2, width//2, 
                            3,4),
                            dtype=img_raw.dtype)
    for j in range(2):
        for i in range(2):
            # (i,j)
            # (0,0) is 90, (0,1) is 45
            # (1,0) is 135, (1,1) is 0

            # Downsampling by 2
            img_bayer_ij = img_raw[i::2, j::2]
            
            # Color correction
            # img_bayer_cc = np.clip(apply_cc_bayer(img_bayer_ij,
            #                               'data/PMVIR_processed/ccmat.mat'),
            #                               0,1)

            # Convert images to 16 bit
            img_bayer_16b = (img_bayer_ij*(2**16-1)).astype('uint16')
            # Color demosaicking
            img_rgb_ij_16b = cv2.cvtColor(img_bayer_16b,
                                          cv2.COLOR_BayerBG2RGB_EA) # Convert to 16bit and use edge aware demosaicking
            
            # Convert back to float 0, 1
            img_rgb_ij = img_rgb_ij_16b.astype('float32')/(2**16-1)

            # import imageio; imageio.imwrite('viz/pmvir_rgb/image_rgb_ij.exr',img_rgb_ij)            
            # img_rgb_us = rescale(img_rgb_ij, 2,
            #                      anti_aliasing=False,
            #                      multichannel=True)
            img_rgb_us = img_rgb_ij
            # Save as stack
            img_pfa_rgb[:,:,:,2*i+j] = img_rgb_us # 90 45 135 0 h w 3 4
            
    # Upsampling 2
    img_pfa_rgb_cat = np.empty((height, width, 3),dtype = np.float32)
    for j in range(2):
        for i in range(2):
            img_pfa_rgb_cat[i::2,j::2] = img_pfa_rgb[:,:,:,2*i+j]
    rgb_dem_stack = []
    for i in range(3):
        mono_dem = pa.demosaicing(img_pfa_rgb_cat[:,:,i])
        rgb_dem_stack.append(mono_dem)
    rgb_dem = np.stack(rgb_dem_stack,-2)
    return rgb_dem

def view_dop(s0, s1, s2, path):
    dop = np.sqrt(s1**2 + s2**2)/(s0+.0)
    cv2.imwrite(path, convert_u8(255*dop))
