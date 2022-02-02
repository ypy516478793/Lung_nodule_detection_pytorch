import nibabel as nib
import numpy as np
import glob, os
import filecmp


def cmp_data():
    """ compare data in two directories """
    dir1 = "/Users/pyuan/PycharmProjects/Lung_nodule_detection_pytorch/Methodist_incidental/data_Kelvin/Nifti_data_old"
    dir2 = "/Users/pyuan/PycharmProjects/Lung_nodule_detection_pytorch/Methodist_incidental/data_Kelvin/Nifti_data"

    data_ls1 = glob.glob(dir1 + "/*/*.nii.gz")
    data_ls2 = glob.glob(dir2 + "/*/*.nii.gz")

    data_ls1.sort()
    data_ls2.sort()

    diff_ls = []
    for i, j in zip(data_ls1, data_ls2):
        if not filecmp.cmp(i, j):
            diff_ls.append(j)

    diff_names = [os.path.basename(i) for i in diff_ls]
    print(diff_names)

def correct_mask_value():
    """ change lung value from 2 to 1 """
    mask_dir = "./Methodist_incidental/data_Kelvin/Lung_masks/Masks_0.6"
    mask_list = glob.glob(mask_dir + "/*/*.nii.gz")  ## all masks
    for mask_path in mask_list:
        hdr = nib.load(mask_path)
        masks = hdr.get_fdata()

        masks = (masks == 2).astype(np.int8)

        masks_nii = nib.Nifti1Image(masks, hdr.affine)
        nib.save(masks_nii, mask_path)
        print("Save to: ", mask_path)

if __name__ == '__main__':
    correct_mask_value()