

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


raw_files = sorted_nicely(glob.glob(dpath + 'Batch*train/Brain*rgt.nii.gz'))

brain_array_sum = np.zeros((512, 512, 32))

# get the summation of all 100 training samples
# To guarantee the symmetricity of the template,
# all training data was sum twice (with additional flipped version)

for i in tqdm(range(100)):
    f = files[i]
    img = nib.load(f)
    img_data = img.get_data()
    brain_array_sum = brain_array_sum + img_data + np.flip(img_data, axis = 0)

# Get averaged tmplate

brain_array_avg = brain_array/200


# set the affine, in-plane resolution 0.426*0.426 and slice thickness 5mm

ref_affine = np.zeros((3,3))
ref_affine[0, 0] = -0.426
ref_affine[1, 1] = 0.426
ref_affine[2, 2] = 5
ref_affine[1, 3] = 0
ref_affine[2, 3] = 0


save_img = nib.Nifti1Image(brain_array_avg, affine=ref_affine)
nib.save(save_img, dpath + 'Templat_5mm_avg.nii.gz')






