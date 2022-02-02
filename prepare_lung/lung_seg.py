from prepare_lung.create_lung_mask import lung_mask_3D
from utils.data_utils import load_nii_resample
from tqdm import tqdm
import nibabel as nib
import pandas as pd
import numpy as np
import os, glob

def lung_seg_nifti(data_dir, mask_save_dir, save_dir, data_path=None):
    """ Lung mask generation for all scans in Nifti format """
    if data_path is not None:
        data_list = [data_path]  ## single scan
    else:
        data_list = glob.glob(data_dir+"/*/*.nii.gz")  ## all scans in data_dir

    lung_not_detected = []
    ## Generate all nifti masks
    for data_path in tqdm(data_list):
        mask_save_path = data_path.replace(data_dir, mask_save_dir).replace(".nii.gz", "_mask.nii.gz")
        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
        if os.path.exists(mask_save_path):
            continue

        hdr = nib.load(data_path)
        images = hdr.get_fdata()
        # masks = lung_mask_2D(images, format="nifti")  ## Old segmentation
        masks = lung_mask_3D(images, max_ind=max_ind, format="nifti")  ## New segmentation
        if masks is None:
            print("no lung for {:s}".format(data_path))
            lung_not_detected.append(data_path)
            continue
        masks_nii = nib.Nifti1Image(masks, hdr.affine)
        nib.save(masks_nii, mask_save_path)
        print("Save to: ", mask_save_path)

    ## Generate all masked data
    for data_path in tqdm(data_list):
        save_path = data_path.replace(data_dir, save_dir)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        dirname = os.path.dirname(save_path)
        filename = os.path.basename(save_path).rstrip(".nii.gz")

        # skip if it has been processed
        if os.path.exists(os.path.join(dirname, filename + "_label.npz")):
            continue

        ## processing image
        images, spacing = load_nii_resample(data_path)  ## resample to [1, 1, 1]
        if not os.path.exists(os.path.join(dirname, filename + '_mask.npz')):
            mask_path = data_path.replace(data_dir, mask_save_dir).replace(".nii.gz", "_mask.nii.gz")
            masks, _ = load_nii_resample(mask_path) ## resample to [1, 1, 1]
            masks = (masks >= 0.5).astype(np.int8)
        else:
            masks = np.load(os.path.join(dirname, filename + '_mask.npz'))["masks"]

        masked_images = masks * images + (1 - masks).astype('uint8') * pad_value
        zz, yy, xx = np.where(masks)
        box = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)], [np.min(xx), np.max(xx)]])
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack([np.max([[0, 0, 0], box[:, 0] - margin], 0),
                               np.min([masks.shape, box[:, 1] + 2 * margin], axis=0).T]).T
        sliceim = masked_images[extendbox[0, 0]:extendbox[0, 1],
                  extendbox[1, 0]:extendbox[1, 1],
                  extendbox[2, 0]:extendbox[2, 1]]
        sliceim = sliceim[np.newaxis, ...]

        ## processing label
        pos_df = pd.read_csv(annot_file)
        pstr, dstr = filename.split("_")
        existId = (pos_df["patient"] == pstr) & (pos_df["date"] == int(dstr))
        label = pos_df[existId][["z", "y", "x", "d"]].values

        label[:, :3] = label[:, :3] * spacing
        isflip = False
        if isflip:
            label[:, 2] = masked_images.shape[2] - label[:, 2]
        label[:, 3] = label[:, 3] * spacing[1]

        if len(label) == 0:
            label = np.array([[0, 0, 0, 0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)
            label = label2[:4].T

        origin = np.array([0, 0, 0])
        np.savez_compressed(os.path.join(dirname, filename + '_clean.npz'), image=sliceim)
        np.savez_compressed(os.path.join(dirname, filename + "_label.npz"), label=label)
        np.savez_compressed(os.path.join(dirname, filename + '_spacing.npz'), spacing=spacing)
        np.savez_compressed(os.path.join(dirname, filename + '_extendbox.npz'), extendbox=extendbox)
        np.savez_compressed(os.path.join(dirname, filename + '_origin.npz'), origin=origin)
        np.savez_compressed(os.path.join(dirname, filename + '_mask.npz'), masks=masks)
        print("save to {:s}".format(os.path.join(dirname, filename + '_clean.npz')))

    print("No lung detected in: ")
    for i in lung_not_detected:
        print(i)


if __name__ == '__main__':
    ## Masked data generalization (Nifti image -> Nifti mask -> Npz masked image)
    data_dir = "./Methodist_incidental/data_Kelvin/Nifti_data"
    mask_save_dir = "./Methodist_incidental/data_Kelvin/Lung_masks/Masks_0.6"
    save_dir = "./Methodist_incidental/data_Kelvin/Masked_data"
    annot_file = "./Methodist_incidental/data_Kelvin/Nifti_data/pos_labels_raw.csv"
    pad_value = -3000

    # data_path = "./Methodist_incidental/data_Kelvin/Nifti_data/Lung_patient002/patient002_20110314.nii.gz"
    # data_path = "./Methodist_incidental/data_Kelvin/Nifti_data/Lung_patient254/patient254_20160613.nii.gz"
    # data_path = "./Methodist_incidental/data_Kelvin/Nifti_data/Lung_patient175/patient175_20150818.nii.gz"
    # data_path = "./Methodist_incidental/data_Kelvin/Nifti_data/Lung_patient373/patient373_20171120.nii.gz"
    data_path = None
    # max_ind = 132
    max_ind = None
    lung_seg_nifti(data_dir, mask_save_dir, save_dir, data_path=data_path)

    # ## Mask generalization (Nifti image -> Nifti mask)
    # data_dir = "./Methodist_incidental/data_Kelvin/Nifti_data"
    # mask_save_dir = "./Methodist_incidental/data_Kelvin/Masked_data/Masks_0.6"
    #
    # data_path = None
    # lung_seg(data_dir, mask_save_dir, data_path=data_path)




