import nibabel as nib
import numpy as np
import glob, os

data_dir = "./Methodist_incidental/data_Ben/resampled"
save_dir = "./Methodist_incidental/data_Ben/resampled_nifti"

data_pattern = "/*/*.npz"

filepaths = glob.glob(data_dir+data_pattern)
filepaths.sort()
filenames = [i.split("/")[-1].rstrip(".npz") for i in filepaths]


for filepath in filepaths:
    tmp = np.load(filepath, allow_pickle=True)
    image, info = tmp["image"], tmp["info"].item()
    image = image.transpose(2, 1, 0)
    image = image[::-1, ::-1, ::-1]
    affine = np.zeros((4, 4))
    affine_diag = [float(i) for i in info["pixelSpacing"]] + [1., 1.]
    # affine_diag[0] = affine_diag[0] * -1
    np.fill_diagonal(affine, affine_diag)

    save_path = os.path.join(filepath.replace(data_dir, save_dir).replace(".npz", ".nii.gz"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image_nii = nib.Nifti1Image(image, affine)
    nib.save(image_nii, save_path)
    print("Save to: ", save_path)


def npz2nii(filepath):
    pixelSpace = 1. #0.8945
    tmp = np.load(filepath, allow_pickle=True)
    if "image" in tmp:
        image = tmp["image"]
        if len(image.shape) == 4 and len(image) == 1: # image.shape == (1, z, y, x)
            image = image[0]
    elif "masks" in tmp:
        image = tmp["masks"]
    else:
        raise ValueError("no image and masks in:", list(tmp.keys()))
    image = image.transpose(2, 1, 0)
    image = image[::-1, ::-1, ::-1].astype(np.float32)
    affine = np.zeros((4, 4))
    affine_diag = [pixelSpace, pixelSpace] + [1., 1.]
    # affine_diag[0] = affine_diag[0] * -1
    np.fill_diagonal(affine, affine_diag)

    save_path = os.path.join(filepath.replace(".npz", ".nii.gz"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image_nii = nib.Nifti1Image(image, affine)
    nib.save(image_nii, save_path)
    print("Save to: ", save_path)


if __name__ == '__main__':
    filepath = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/resampled/Lung_patient468/patient468-20180625.npz"
    npz2nii(filepath)

# filepath = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Kelvin/Lung_patient002/patient002_20090310.nii.gz"
# hdr = nib.load(filepath)
# CT = hdr.get_fdata()
# vox_size = np.abs(hdr.affine.diagonal()[:3])
#
#
# if len(image) == 1 and len(image.shape) == 4:
#     image = image[0]
#
#
#
# for filepath in filepaths:
#     image = np.load(filepath, allow_pickle=True)["image"]
#     if len(image) == 1 and len(image.shape) == 4:
#         image = image[0]
#
# filepath = os.path.join(data_dir, filename)
# hdr = nib.load(filepath)
# CT = hdr.get_fdata()
# vox_size = np.abs(hdr.affine.diagonal()[:3])
#
#
# save_path = os.path.join(data_dir, filename.replace(".nii.gz", "_mask.nii.gz"))
# mask_nii = nib.Nifti1Image(mask, hdr.affine)
# nib.save(mask_nii, save_path)
# print("Save to: ", save_path)