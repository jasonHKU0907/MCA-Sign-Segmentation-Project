

import numpy as np
import os
import scipy.ndimage
import cv2
import nibabel as nib
import nipype.interfaces.fsl as fsl
import glob
from scipy import ndimage
from tqdm import tqdm
import re

dpath='../'


def sorted_nicely(l):
    '''
    :param l: files
    :return: sorted files
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def threshold(voxels, lower, upper):
    '''
    :param voxels: input voxels
    :param lower: threshold lower bound
    :param upper: threshold upper bound
    :return: threshold voxels
    '''
    voxels_thre = voxels.copy()
    voxels_thre[voxels_thre <= lower] = lower
    voxels_thre[voxels_thre >= upper+1] = upper+1
    voxels_thre[voxels_thre == upper+1] = lower
    return voxels_thre


def largest_cc(mask):
    '''
    :param mask: brain masks
    :return: largest connect component
    '''
    mask = np.asarray(mask)
    labels, label_nb = ndimage.label(mask)
    if not label_nb:
        raise ValueError('No non-zero values: no connected components')
    if label_nb == 1:
        return mask.astype(np.bool)
    label_count = np.bincount(labels.ravel().astype(np.int))
    label_count[0] = 0
    return labels == label_count.argmax()


# First do brain extraction
# 1st threshold brain within HU [0, 100]
# 2nd Binarize threshed scans and fill holes to get a mask (based on the 1st step)
# 3rd Apply gaussian blur to threshed scans (based on the 1st step)
# 4th Multiplies step2 & step3 and save as a smoothed scan
# 5th Apply BET through FSL to extract the brain (based on the 4th step)
# 6th Read the brain, binarize it to get a mask, fill holes and do erosion
# 7th Find the largest connected component, fill holes
#     Step6 & step7 are aimed to remove those edges that BET not performed well
# 8th Finally multiplies the mask achieved from 7th step and original CT scans


raw_files = sorted_nicely(glob.glob(dpath + 'Batch*/Batch*.nii.gz'))

nb_brains = len(raw_files)

for i in tqdm(range(nb_brains)):
    # load nii file and read into numpy array
    f = raw_files[i]
    cur_dir = os.path.dirname(f)
    basename = os.path.basename(f)
    img = nib.load(f)
    img_hdr = img.header
    img_pixels = img.get_data().astype(np.int16)
    img_thr0 = img_pixels.copy()
    img_thr0 = img_thr0

    # step 1
    img_thr0 = threshold(img_thr0, lower=0, upper=100)

    # step 2
    img_bin = cv2.threshold(img_thr0, 0, 1, cv2.THRESH_BINARY)[1]
    img_fill = scipy.ndimage.morphology.binary_fill_holes(img_bin)

    # step 3
    img_sm = cv2.GaussianBlur(img_thr0, (9, 9), 9)

    # step 4
    img_sm_masked = img_sm * img_fill
    save_img = nib.Nifti1Image(img_sm_masked, affine=img.affine)
    nib.save(save_img, dpath + 'img_sm.nii.gz')

    # step 5
    mybet = fsl.BET()
    mybet.inputs.in_file = dpath + 'img_sm.nii.gz'
    mybet.inputs.out_file = dpath + 'img_bet.nii.gz'
    mybet.inputs.frac = 0.01
    mybet.inputs.robust = True
    mybet.inputs.output_type = "NIFTI_GZ"
    result = mybet.run()

    # step 6
    img_bet = nib.load(dpath + 'img_bet.nii.gz')
    img_bet_pixels = img_bet.get_data().astype(np.int16)
    img_bet_bin = cv2.threshold(img_bet_pixels, 0, 1, cv2.THRESH_BINARY)[1]
    img_bet_mask = np.transpose(img_bet_bin, (2, 0, 1))
    for i in range(len(img_bet_mask)):
        img_bet_mask[i] = scipy.ndimage.morphology.binary_fill_holes(img_bet_mask[i])
    img_bet_mask = np.transpose(img_bet_mask, (1, 2, 0))
    img_final_msk = ndimage.binary_erosion(img_bet_mask).astype(img_bet_mask.dtype)

    # step 7
    img_final_msk = largest_cc(img_final_msk)
    img_final_msk = np.transpose(img_final_msk, (2, 0, 1))
    for i in range(len(img_final_msk)):
        img_final_msk[i] = scipy.ndimage.morphology.binary_fill_holes(img_final_msk[i])
    img_final_msk = np.transpose(img_final_msk, (1, 2, 0))

    # step 8
    img_preprocessed = img_pixels * img_final_msk
    save_img = nib.Nifti1Image(img_preprocessed, affine=img.affine, header = img_hdr)
    nib.save(save_img, cur_dir + '/Brain' + basename[5:])


