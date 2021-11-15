
from scipy.ndimage import binary_fill_holes
from collections import Counter
from skimage import measure
import nibabel as nib
import numpy as np
import os

data_dir = "./Methodist_incidental"
filename = "20160716_CT.nii.gz"

filepath = os.path.join(data_dir, filename)
hdr = nib.load(filepath)
CT = hdr.get_fdata()
vox_size = np.abs(hdr.affine.diagonal()[:3])

Fat=-400
mask=CT>Fat

CC = measure.label(mask)
numPixels = Counter(CC[CC != 0])
idx, biggest = numPixels.most_common(1)[0]
mask = np.zeros_like(CT)
mask[CC == idx] = 1

# Assume 40% threshold for body cross section is neck position
# ind=find(head_to_toe_profile>.4*max(head_to_toe_profile));
head_to_toe_profile = np.sum(mask, axis=(0, 1))
# Assume 40% threshold for body cross section is neck position
ind = np.where(head_to_toe_profile > 0.4 * max(head_to_toe_profile))[0]
# Smart masking of neck and up
mask[1: -1, 1: -1, : min(ind) + 1] = 1
mask[1: -1, 1: -1, max(ind): ] = 1


CC = measure.label(1 - mask)
numPixels = Counter(CC[CC != 0])
idx = sorted(numPixels.keys(), key=numPixels.get, reverse=True)

# identify background
for i in range(len(idx)):
    if 0 in np.arange(CC.size)[(CC == idx[i]).reshape(-1)]:
        mask[CC == idx[i]] = -1
        print(i)
        continue
    if i <= 1:
        mask[CC == idx[i]] = 2
        continue
    else:
        break


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


# hdr.fname='20160716_CT_mask.nii'
# hdr.dt=[spm_type('int8') 0]
# spm_write_vol(hdr,mask)
# gzip(hdr.fname)
# delete(hdr.fname)



print("")