
from scipy.ndimage import binary_fill_holes
from collections import Counter
from skimage import measure
from tqdm import tqdm
import nibabel as nib
import numpy as np
import os, glob

def lung_seg_single(images, max_ind=None):
    Fat = -400
    mask = images > Fat

    CC = measure.label(mask, connectivity=2)
    ## Make sure the first layer and the last layer is the background has the background
    if not 0 in CC[:, :, 0]:
        CC[:, :, 0] = 0
    if not 0 in CC[:, :, -1]:
        CC[:, :, -1] = 0
    # CC = measure.label(mask)
    numPixels = Counter(CC[CC != 0])
    idx, biggest = numPixels.most_common(1)[0]
    mask = np.zeros_like(images)
    mask[CC == idx] = 1

    # Assume 40% threshold for body cross section is neck position
    # ind=find(head_to_toe_profile>.4*max(head_to_toe_profile));
    head_to_toe_profile = np.sum(mask, axis=(0, 1))
    # Assume 40% threshold for body cross section is neck position
    max_area = max(head_to_toe_profile[head_to_toe_profile != mask.shape[0] * mask.shape[1]])
    ind = np.where(head_to_toe_profile > 0.7 * max_area)[0]  # change from 0.4 to 0.6, to 0.7
    # Smart masking of neck and up
    mask[1: -1, 1: -1, : min(ind) + 1] = 1
    if max_ind is not None:
        mask[1: -1, 1: -1, max_ind:] = 1
    else:
        mask[1: -1, 1: -1, max(ind):] = 1
    CC = measure.label(1 - mask, connectivity=2)
    # CC = measure.label(1 - mask)
    numPixels = Counter(CC[CC != 0])
    idx = sorted(numPixels.keys(), key=numPixels.get, reverse=True)

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
                    locs[0].max() > x / 2
                    # locs[1].min() < y / 2 and
                    # locs[1].max() > y / 2
            ):
                mask[CC == idx[i]] = 2  # lung
                find_lung = True
            else:
                mask[CC == idx[i]] = 1  # other area
        if i >= 5:
            return None # no lung is found


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

    return mask


def lung_seg(data_dir, save_dir):
    file_list = glob.glob(data_dir+"/*/*.nii.gz")
    for filepath in tqdm(file_list):
        # filepath = os.path.join(data_dir, filename)
        hdr = nib.load(filepath)
        CT = hdr.get_fdata()
        vox_size = np.abs(hdr.affine.diagonal()[:3])

        mask = lung_seg_single(CT)

        # Fat=-400
        # mask=CT>Fat
        #
        # CC = measure.label(mask, connectivity=2)
        # ## Make sure the first layer and the last layer is the background has the background
        # if not 0 in CC[:, :, 0]:
        #     CC[:, :, 0] = 0
        # if not 0 in CC[:, :, -1]:
        #     CC[:, :, -1] = 0
        # # CC = measure.label(mask)
        # numPixels = Counter(CC[CC != 0])
        # idx, biggest = numPixels.most_common(1)[0]
        # mask = np.zeros_like(CT)
        # mask[CC == idx] = 1
        #
        # # Assume 40% threshold for body cross section is neck position
        # # ind=find(head_to_toe_profile>.4*max(head_to_toe_profile));
        # head_to_toe_profile = np.sum(mask, axis=(0, 1))
        # # Assume 40% threshold for body cross section is neck position
        # max_area = max(head_to_toe_profile[head_to_toe_profile != mask.shape[0] * mask.shape[1]])
        # ind = np.where(head_to_toe_profile > 0.6 * max_area)[0]  # change from 0.4 to 0.6
        # # Smart masking of neck and up
        # mask[1: -1, 1: -1, : min(ind) + 1] = 1
        # mask[1: -1, 1: -1, max(ind): ] = 1
        #
        #
        # CC = measure.label(1 - mask, connectivity=2)
        # # CC = measure.label(1 - mask)
        # numPixels = Counter(CC[CC != 0])
        # idx = sorted(numPixels.keys(), key=numPixels.get, reverse=True)
        #
        # # # identify background
        # # for i in range(len(idx)):
        # #     if 0 in np.arange(CC.size)[(CC == idx[i]).reshape(-1)]:
        # #         mask[CC == idx[i]] = -1
        # #         print(i)
        # #         continue
        # #     if i <= 1:
        # #         mask[CC == idx[i]] = 2
        # #         continue
        # #     else:
        # #         break
        #
        # x, y, z = CC.shape
        # # identify background
        # find_lung = False
        # find_bgd = False
        # for i in range(len(idx)):
        #     if find_lung:
        #         break
        #     if not find_bgd and 0 in np.arange(CC.size)[(CC == idx[i]).reshape(-1)]:
        #         find_bgd = True
        #         mask[CC == idx[i]] = -1  # back_ground
        #     else:
        #         locs = np.where(CC == idx[i])
        #         if (
        #                 locs[0].min() < x / 2 and
        #                 locs[0].max() > x / 2 and
        #                 locs[1].min() < y / 2 and
        #                 locs[1].max() > y / 2
        #         ):
        #             mask[CC == idx[i]] = 2  # lung
        #             find_lung = True
        #         else:
        #             mask[CC == idx[i]] = 1  # other area
        #
        # # Fill Sagittal
        # mask1 = (mask == 2)
        # for i in range(mask1.shape[1]):
        #     tmp = mask1[:, i, :]
        #     tmp = binary_fill_holes(tmp)
        #     mask[:, i, :] = 2 * tmp
        #
        # # Fill Coronal
        # mask1 = (mask == 2)
        # for i in range(mask1.shape[0]):
        #     tmp = mask1[i, :, :]
        #     tmp = binary_fill_holes(tmp)
        #     mask[i, :, :] = 2 * tmp
        #
        # # Fill Axial
        # mask1 = (mask == 2)
        # for i in range(mask1.shape[2]):
        #     tmp = mask1[:, :, i]
        #     tmp = binary_fill_holes(tmp)
        #     mask[:, :, i] = 2 * tmp

        save_path = filepath.replace(data_dir, save_dir).replace(".nii.gz", "_mask.nii.gz")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # save_path = os.path.join(data_dir, filename.replace(".nii.gz", "_mask.nii.gz"))
        mask_nii = nib.Nifti1Image(mask, hdr.affine)
        nib.save(mask_nii, save_path)
        print("Save to: ", save_path)


import pandas as pd

def mask_scan_Kelvin(images):
    # Reverse the image from [z, y, x] to [x, y, z]
    images = images.transpose(2, 1, 0)
    images = images[::-1, ::-1, ::-1]

    pad_value = -3000
    masks = lung_seg_single(images, max_ind=max_ind)
    if masks is None:
        return None, None
    masks = (masks == 2).astype(np.int)

    # Reverse the image back from [x, y, z] to [z, y, x]
    masks = masks.transpose(2, 1, 0)
    masks = masks[::-1, ::-1, ::-1]
    images = images.transpose(2, 1, 0)
    images = images[::-1, ::-1, ::-1]

    masked_images = masks * images + (1 - masks).astype('uint8') * pad_value

    return masked_images, masks


def lung_seg_npz():
    save_path = data_path.replace(root_dir, save_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dirname = os.path.dirname(save_path)
    filename = os.path.basename(save_path).rstrip(".npz")
    if os.path.exists(os.path.join(dirname, filename + '_label.npz')):
        return

    imgs = np.load(data_path, allow_pickle=True)["image"]
    # imgs, masks = mask_scan(imgs)  ## Old segmentation
    imgs, masks = mask_scan_Kelvin(imgs)  ## New segmentation
    if masks is None:
        print("no lung for {:s}".format(data_path))
        return
    # imgs = lumTrans(imgs)

    zz, yy, xx = np.where(masks)
    box = np.array([[np.min(zz), np.max(zz)], [np.min(yy), np.max(yy)], [np.min(xx), np.max(xx)]])
    box = np.floor(box).astype('int')
    margin = 5
    extendbox = np.vstack([np.max([[0, 0, 0], box[:, 0] - margin], 0),
                           np.min([masks.shape, box[:, 1] + 2 * margin], axis=0).T]).T
    sliceim = imgs[extendbox[0, 0]:extendbox[0, 1],
                   extendbox[1, 0]:extendbox[1, 1],
                   extendbox[2, 0]:extendbox[2, 1]]
    sliceim = sliceim[np.newaxis, ...]

    annot_file = "./Methodist_incidental/data_Ben/resampled/pos_labels_norm.csv"
    pos_df = pd.read_csv(annot_file)

    pstr, dstr = filename.split("-")
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
    np.savez_compressed(os.path.join(dirname, filename+'_clean.npz'), image=sliceim)
    # np.save(os.path.join(dirname, filename+'_spacing.npy'), spacing)
    np.savez_compressed(os.path.join(dirname, filename+'_extendbox.npz'), extendbox=extendbox)
    # np.save(os.path.join(dirname, filename+'_origin.npy'), origin)
    np.savez_compressed(os.path.join(dirname, filename+'_mask.npz'), masks=masks)
    print("save to {:s}".format(os.path.join(dirname, filename+'_clean.npz')))



if __name__ == '__main__':
    # data_dir = "./Methodist_incidental/data_Kelvin/Nifti_data"
    # save_dir = "./Methodist_incidental/data_Kelvin/Masked_data/Masks_0.6"
    # lung_seg(data_dir, save_dir)

    root_dir = "Methodist_incidental/data_Ben/resampled/"
    save_dir = "Methodist_incidental/data_Ben/masked_data_v1/"
    data_path = "Methodist_incidental/data_Ben/resampled/Lung_patient175/patient175-20150818.npz"
    max_ind = 332
    lung_seg_npz()
# hdr.fname='20160716_CT_mask.nii'
# hdr.dt=[spm_type('int8') 0]
# spm_write_vol(hdr,mask)
# gzip(hdr.fname)
# delete(hdr.fname)


