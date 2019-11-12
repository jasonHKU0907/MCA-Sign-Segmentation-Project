


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




# Read each scan and register them
# Store registered scans into Registered folder
# Store transformation matrices into Registered_Matrix folder
# Transform raw masks and stored in Registered_Masks folder

files = sorted_nicely(glob.glob(dpath + 'Batch*/Brain*.nii.gz'))

reference = dpath + 'Templat_5mm_avg.nii.gz'


for i in tqdm(range(len(files))):
    flt = fsl.FLIRT(bins=640, cost_func='corratio')
    flt.inputs.in_file = files[i]
    basename = os.path.basename(files[i])
    cur_dir = os.path.dirname(files[i])
    flt.inputs.reference = reference
    flt.inputs.out_file = cur_dir + '/' + basename[:-7] + '_rgt.nii.gz'
    flt.inputs.out_matrix_file = cur_dir + '/' + basename[:-7] + '_rgt.mat'
    flt.inputs.interp = 'trilinear'
    flt.inputs.dof = 12
    flt.inputs.output_type = "NIFTI_GZ"
    #flt.cmdline
    res = flt.run()


tmp = 'registered_01_01.nii.gz'
tmp[:-7]


Masks = sorted_nicely(glob.glob(dpath + 'Batch*/Mask*.nii.gz'))
Matrix = sorted_nicely(glob.glob(dpath + 'Batch*/Brain*rgt.mat'))
References = sorted_nicely(glob.glob(dpath + 'Batch*/Brain*rgt.mat'))

len(References)

for i in tqdm(range(len(Masks))):
    mask = Masks[i]
    basename = os.path.basename(mask)
    cur_dir = os.path.dirname(mask)
    flt = fsl.FLIRT(bins=640, cost_func='corratio')
    flt.inputs.in_file = mask
    flt.inputs.in_matrix_file = Matrix[i]
    flt.inputs.out_file = cur_dir + '/' + basename[:-7] + '_mask.nii.gz'
    flt.inputs.reference = References[i]
    flt.inputs.apply_xfm = True
    flt.inputs.output_type = "NIFTI_GZ"
    #flt.cmdline
    result = flt.run()
    msk = nib.load(cur_dir + '/' + basename[:-7] + '_mask.nii.gz')
    msk_affine = msk._affine
    msk_hdr = msk.header
    msk_pixels = msk.get_data()
    msk_pixels[msk_pixels >= 0.5] = 1
    msk_pixels[msk_pixels < 0.5] = 0
    img = nib.Nifti1Image(msk_pixels, msk_affine, header=msk_hdr)
    nib.save(img, cur_dir + '/' + basename[:-7] + '_mask.nii.gz')






