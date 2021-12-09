
from scipy.ndimage import binary_fill_holes
from collections import Counter
from skimage import measure
import nibabel as nib
import numpy as np
import os
import time

start = time.time()

# data_dir = "./Methodist_incidental/lung_segmentation_debug"
# data_dir = "./Methodist_incidental/data_Ben/resampled_nifti/Lung_patient468/"
# data_dir = "./Methodist_incidental/data_Ben/resampled_nifti/Lung_patient005/"
# data_dir = "./Methodist_incidental/data_Ben/resampled_nifti/Lung_patient413"
# data_dir = "./Methodist_incidental/data_Ben/resampled_nifti/Lung_patient175"
# data_dir = "./Methodist_incidental/data_Ben/resampled_nifti/Lung_patient453/"
# data_dir = "./Methodist_incidental/data_Ben/resampled_nifti/Lung_patient455/"
# data_dir = "./Methodist_incidental/data_Ben/resampled_nifti/Lung_patient422/"
data_dir = "./Methodist_incidental/data_Ben/resampled_nifti/Lung_patient373/"
# filename = "20160716_CT.nii.gz"
# filename = "20130912_PET-CT.nii.gz"
# filename = "patient477_20181017.nii.gz"
# filename = "patient475_20180928.nii.gz"
# filename = "patient237_20160725.nii.gz"
# filename = "patient413-20180409.nii.gz"
# filename = "patient005-20120524.nii.gz"
# filename = "patient175-20150818.nii.gz"
# filename = "patient453-20180530.nii.gz"
# filename = "patient455-20180720.nii.gz"
# filename = "patient422-20180507.nii.gz"
filename = "patient373-20171120.nii.gz"



filepath = os.path.join(data_dir, filename)
hdr = nib.load(filepath)
CT = hdr.get_fdata()
vox_size = np.abs(hdr.affine.diagonal()[:3])

Fat=-400
mask=CT>Fat

CC = measure.label(mask, connectivity=2)
## Make sure the first layer and the last layer is the background has the background
if not 0 in CC[: ,: ,0]:
    CC[: ,: ,0] = 0
if not 0 in CC[: ,: ,-1]:
    CC[: ,: ,-1] = 0
# CC = measure.label(mask)
tmp = CC[CC != 0]
numPixels = Counter(tmp)
idx, biggest = numPixels.most_common(1)[0]
mask = np.zeros_like(CT)
mask[CC == idx] = 1

save_path = os.path.join(data_dir, filename.replace(".nii.gz", "_mask_step1.nii.gz"))
mask_nii = nib.Nifti1Image(mask, hdr.affine)
nib.save(mask_nii, save_path)
print("Save to: ", save_path)

# Assume 40% threshold for body cross section is neck position
# ind=find(head_to_toe_profile>.4*max(head_to_toe_profile));
head_to_toe_profile = np.sum(mask, axis=(0, 1))
# Assume 40% threshold for body cross section is neck position
max_area = max(head_to_toe_profile[head_to_toe_profile != mask.shape[0] * mask.shape[1]])
ind = np.where(head_to_toe_profile > 0.7 * max_area)[0]  # change from 0.4 to 0.6
# Smart masking of neck and up

max_ind = 332
mask[1: -1, 1: -1, : min(ind) + 1] = 1
mask[1: -1, 1: -1, max(ind): ] = 1
# mask[1: -1, 1: -1, max_ind: ] = 1

save_path = os.path.join(data_dir, filename.replace(".nii.gz", "_mask_step2.nii.gz"))
mask_nii = nib.Nifti1Image(mask, hdr.affine)
nib.save(mask_nii, save_path)
print("Save to: ", save_path)


CC = measure.label(1 - mask, connectivity=2)
# CC = measure.label(1 - mask)
tmp = CC[CC != 0]
numPixels = Counter(tmp)
idx = sorted(numPixels.keys(), key=numPixels.get, reverse=True)

# # identify background (old)
# for i in range(len(idx)):
#     if 0 in np.arange(CC.size)[(CC == idx[i]).reshape(-1)]:
#         mask[CC == idx[i]] = -1
#         print(i)
#         continue
#     if i <= 1:
#         mask[CC == idx[i]] = 2
#         continue
#     else:
#         break

x, y, z = CC.shape
# identify background
find_lung = False
find_bgd = False
for i in range(len(idx)):
    if find_lung:
        break
    if not find_bgd and 0 in np.arange(CC.size)[(CC == idx[i]).reshape(-1)]:
        find_bgd = True
        mask[CC == idx[i]] = -1  # back_ground
    else:
        locs = np.where(CC == idx[i])
        if (
            locs[0].min() < x / 2 and
            locs[0].max() > x / 2 and
            locs[1].min() < y / 2 and
            locs[1].max() > y / 2
        ):
            mask[CC == idx[i]] = 2  # lung
            find_lung = True
        else:
            mask[CC == idx[i]] = 1  # other area
    if i >= 5:
        print("No lung is found!")
        break

save_path = os.path.join(data_dir, filename.replace(".nii.gz", "_mask_step3.nii.gz"))
mask_nii = nib.Nifti1Image(mask, hdr.affine)
nib.save(mask_nii, save_path)
print("Save to: ", save_path)

# Fill Sagittal
mask1 = (mask == 2)
for i in range(mask1.shape[1]):
    tmp = mask1[:, i, :]
    tmp = binary_fill_holes(tmp)
    mask[:, i, :] = 2 * tmp

# Fill Coronal
mask1 = (mask == 2)
for i in range(mask1.shape[0]):
    tmp = mask1[i, :, :]
    tmp = binary_fill_holes(tmp)
    mask[i, :, :] = 2 * tmp

# Fill Axial
mask1 = (mask == 2)
for i in range(mask1.shape[2]):
    tmp = mask1[:, :, i]
    tmp = binary_fill_holes(tmp)
    mask[:, :, i] = 2 * tmp


save_path = os.path.join(data_dir, filename.replace(".nii.gz", "_mask.nii.gz"))
mask_nii = nib.Nifti1Image(mask, hdr.affine)
nib.save(mask_nii, save_path)
print("Save to: ", save_path)

print("Spent: {:f}".format(time.time() - start))
