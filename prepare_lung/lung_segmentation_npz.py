from .create_lung_mask import lung_mask_3D
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob, os

def lung_seg_npz(data_dir, save_dir, data_path):
    if data_path is not None:
        data_ls = [data_path]  ## single scan
    else:
        data_ls = glob.glob(data_dir+"/*/*.npz")  ## all scans in data_dir

    lung_not_detected = []
    for data_path in tqdm(data_ls):

        save_path = data_path.replace(data_dir, save_dir)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        dirname = os.path.dirname(save_path)
        filename = os.path.basename(save_path).rstrip(".npz")

        # skip if it has been processed
        if os.path.exists(os.path.join(dirname, filename + "_label.npz")):
            continue

        ## processing image
        images = np.load(data_path, allow_pickle=True)["image"]
        if not os.path.exists(os.path.join(dirname, filename + '_mask.npz')):
            # masks = lung_mask_2D(images, format="npz")  ## Old segmentation
            masks = lung_mask_3D(images, max_ind=max_ind, format="npz")  ## New segmentation
        else:
            masks = np.load(os.path.join(dirname, filename + '_mask.npz'))["masks"]

        if masks is None:
            print("no lung for {:s}".format(data_path))
            lung_not_detected.append(data_path)
            continue

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
        pstr, dstr = filename.split("-")
        existId = (pos_df["patient"] == pstr) & (pos_df["date"] == int(dstr))
        label = pos_df[existId][["z", "y", "x", "d"]].values

        if len(label) == 0:
            label = np.array([[0, 0, 0, 0]])
        else:
            label2 = np.copy(label).T
            label2[:3] = label2[:3] - np.expand_dims(extendbox[:, 0], 1)
            label = label2[:4].T

        spacing = np.array([1, 1, 1])
        origin = np.array([0, 0, 0])
        np.savez_compressed(os.path.join(dirname, filename + '_clean.npz'), image=sliceim)
        np.savez_compressed(os.path.join(dirname, filename + "_label.npz"), label=label)
        np.savez_compressed(os.path.join(dirname, filename+'_spacing.npz'), spacing=spacing)
        np.savez_compressed(os.path.join(dirname, filename+'_extendbox.npz'), extendbox=extendbox)
        np.savez_compressed(os.path.join(dirname, filename+'_origin.npz'), origin=origin)
        np.savez_compressed(os.path.join(dirname, filename+'_mask.npz'), masks=masks)
        print("save to {:s}".format(os.path.join(dirname, filename+'_clean.npz')))

    print("No lung detected in: ")
    for i in lung_not_detected:
        print(i)


if __name__ == '__main__':
    ## Masked data generalization (Npz image -> Npz masked image)
    data_dir = "./Methodist_incidental/data_Ben/resampled/"
    # save_dir = "./Methodist_incidental/data_Ben/masked/"  # use lung_mask_2D
    save_dir = "./Methodist_incidental/data_Ben/masked_data_v1/"  # use lung_mask_3D
    annot_file = "./Methodist_incidental/data_Ben/resampled/pos_labels_norm.csv"
    pad_value = -3000
    
    # data_path = "Methodist_incidental/data_Ben/resampled/Lung_patient175/patient175-20150818.npz"
    # data_path = "Methodist_incidental/data_Ben/resampled/Lung_patient383/patient383-20180124.npz"
    # data_path = "Methodist_incidental/data_Ben/resampled/Lung_patient427/patient427-20180430.npz"
    # data_path = "Methodist_incidental/data_Ben/resampled/Lung_patient354/patient354-20171013.npz"
    # data_path = "Methodist_incidental/data_Ben/resampled/Lung_patient444/patient444-20180105.npz"
    # data_path = "Methodist_incidental/data_Ben/resampled/Lung_patient422/patient422-20180507.npz"
    # data_path = "Methodist_incidental/data_Ben/resampled/Lung_patient394/patient394-20180209.npz"
    # data_path = "Methodist_incidental/data_Ben/resampled/Lung_patient340/patient340-20170810.npz"
    data_path = None
    max_ind = None

    lung_seg_npz(data_dir, save_dir, data_path)