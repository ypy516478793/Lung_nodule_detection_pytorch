from utils.data_utils import load_nii, load_nii_resample
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt


from tqdm import tqdm
import pandas as pd
import glob

import argparse






def mask_lung(data_path):
    save_path = data_path.replace(data_dir, save_dir).replace(".nii.gz", ".npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dirname = os.path.dirname(save_path)
    filename = os.path.basename(save_path).rstrip(".npz")

    ## processing image
    if os.path.exists(os.path.join(dirname, filename + '_clean.npz')):
        return

    mask_path = data_path.replace(data_dir, mask_dir).replace(".nii.gz", "_mask.nii.gz")
    new_image, spacing = load_nii_resample(data_path)  ## resample to [1, 1, 1]
    new_mask, _ = load_nii_resample(mask_path)
    new_mask = (new_mask > 1).astype(np.int)

    # new_image, new_spacing = resample_image(image, spacing)
    # new_mask, _ = resample_image(mask, spacing)
    # new_mask = (new_mask > 1).astype(np.int)

    imgs = new_mask * new_image + (1 - new_mask).astype('uint8') * pad_value
    masks = new_mask

    zz, yy, xx = np.where(masks)
    box = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)], [np.min(xx), np.max(xx)]])
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0, 0, 0], box[:, 0] - margin], 0),
                           np.min([masks.shape, box[:, 1] + 2 * margin], axis=0).T]).T
    scan = imgs[extendbox[0, 0]:extendbox[0, 1],
              extendbox[1, 0]:extendbox[1, 1],
              extendbox[2, 0]:extendbox[2, 1]]
    scan = scan[np.newaxis, ...]

    ## processing label
    if os.path.exists(os.path.join(dirname, filename + '_label.npz')):
        return



    # slices = np.unique(zs[0]) # slices with annotations
    # for z in slices:
    #
    #
    #
    # annot_file = "./Methodist_incidental/data_Ben/resampled/pos_labels_norm.csv"
    # pos_df = pd.read_csv(annot_file)

    # TODO: fixme: Verify the annotation when use data_Kelvin



    pstr, dstr = filename.split("_")
    patient_colname = "patient" if "patient" in pos_df.columns else 'Patient\n Index'
    assert patient_colname in pos_df
    existId = (pos_df[patient_colname] == pstr) & (pos_df["date"] == int(dstr))
    label = pos_df[existId][["z", "y", "x", "d"]].values

    if len(label) == 0:
        label = np.array([[0, 0, 0, 0]])
    else:
        label2 = np.copy(label).T
        label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)
        label = label2[:4].T

    np.savez_compressed(os.path.join(dirname, filename + "_label.npz"), label=label)

    # dirname = os.path.dirname(save_path)
    # filename = os.path.basename(save_path).strip(".npz")
    # spacing = np.array([1, 1, 1])
    # origin = np.array([0, 0, 0])
    np.savez_compressed(os.path.join(dirname, filename + '_clean.npz'), image=scan)
    np.save(os.path.join(dirname, filename+'_spacing.npz'), spacing=spacing)
    np.savez_compressed(os.path.join(dirname, filename + '_extendbox.npz'), extendbox=extendbox)
    # np.save(os.path.join(dirname, filename+'_origin.npy'), origin)
    np.savez_compressed(os.path.join(dirname, filename + '_mask.npz'), masks=masks)
    print("save to {:s}".format(os.path.join(dirname, filename + '_clean.npz')))


def prepare_masked_Kelvin():
    """
    Use new mask crop function from Kelvin. Nifti image + Nifti mask -> Npz image
    Args:
        data_dir: ./Methodist_incidental/data_Kelvin/Nifti_data
        save_dir: ./Methodist_incidental/data_Kelvin/Masked_data
        mask_dir: ./Methodist_incidental/data_Kelvin/Lung_masks/Masks_0.6
        [data_path: ./Methodist_incidental/data_Kelvin/Nifti_data/Lung_patient002/patient002_20090310.nii.gz (e.g.)]
    Return:
        None
    """

    data_ls = glob.glob(data_dir + "/*/*.nii.gz")
    data_ls.sort()


    ## single process
    for data_path in tqdm(data_ls):
        mask_lung(data_path)

    # # multi process
    # from multiprocessing import Pool
    # pool = Pool(32)
    # pool.map(mask_lung, data_ls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="prepare script")
    parser.add_argument('-s', '--save_dir', type=str, help='save directory', default=None)
    parser.add_argument('-m', '--mask_dir', type=str, help='mask directory', default=None)
    parser.add_argument('-d', '--data_dir', type=str, help='data directory', default=None)
    args = parser.parse_args()

    pad_value = -3000
    data_dir = args.data_dir
    save_dir = args.save_dir
    mask_dir = args.mask_dir
    prepare_masked_Kelvin()