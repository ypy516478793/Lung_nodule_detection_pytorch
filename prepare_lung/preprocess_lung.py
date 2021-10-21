from skimage.filters import threshold_otsu
from show_results import plot_bbox
from statistics import mode
from shutil import copyfile
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# def mode_norm_old(image):
#     # t = threshold_otsu(image)
#     t = -400
#     m = mode(image[image < t])
#     # from collections import Counter
#     # c = Counter(image[image < t])
#     # c.most_common(3)
#     std = np.std(image[image < t])
#     image_norm = (image - m) / std
#     return image_norm

from time import time

def safe_mode(array):
    c = Counter(array)
    most_common = np.array(c.most_common())
    top = [most_common[0, 1]]
    for n in most_common[1:, 1]:
        if n != top[-1]:
            top.append(n)
        if len(top) == 2:
            break

    l1 = most_common[most_common[:, 1] == top[0], 0]
    l2 = most_common[most_common[:, 1] == top[1], 0]

    m = l1[np.argmin(np.abs(l1 - np.mean(l2)))]
    return m


# Non-linear, std = 1 for both stacks
def mode_norm3(scan, pad_value, verbose=False):
    center = 4

    scan[np.logical_and(scan != pad_value, scan < -1200)] = pad_value
    scan[np.logical_and(scan != pad_value, scan > 600)] = pad_value

    t = -400
    first_stack = scan[np.logical_and(scan != pad_value, scan < t)]
    second_stack = scan[np.logical_and(scan != pad_value, scan > t // 2)]

    mass, idx = np.histogram(first_stack, bins=100)
    i = np.argmax(mass)
    assert i != 0
    r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
    r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
    m1 = safe_mode(first_stack)
    std1 = idx[r2] - idx[r1]
    if verbose: print("Mode1: {:f}, std1: {:f}".format(m1, std1))

    mass, idx = np.histogram(second_stack, bins=100)
    i = np.argmax(mass)
    assert i != 0
    r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
    r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
    m2 = safe_mode(second_stack)
    std2 = idx[r2] - idx[r1]
    if verbose: print("Mode2: {:f}, std2: {:f}".format(m2, std2))

    shift = (m1 + m2) / 2
    scale = center * 2 / (m2 - m1)
    if verbose: print("Shift: {:f}, scale: {:f}".format(shift, scale))

    scan = scan.astype(np.float)
    scan[scan == pad_value] = shift
    # scan[scan > 282.35] = shift   ## Remove Bone
    scan = scale * (scan - shift)

    mass, idx = np.histogram(scan[scan < 0], bins=100)
    i = np.argmax(mass)
    r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
    r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
    std1 = idx[r2] - idx[r1]

    m1 = safe_mode(scan[scan < 0])

    # scan_left = (scan[scan != 0] - m1) / std1 + m1
    # scan_left = np.clip(scan_left, None, 0)
    scan_left = scan[scan < 0]
    scan_left = np.clip((scan_left - m1) / std1 + m1, None, 0)
    scan[scan < 0] = scan_left

    mass, idx = np.histogram(scan_left, bins=100)
    i = np.argmax(mass[:-1])
    r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
    r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
    std1 = idx[r2] - idx[r1]

    if verbose: print("After mode normalization: Mode1: {:f}, std1: {:f}".format(m1, std1))

    mass, idx = np.histogram(scan[scan > 0], bins=100)
    i = np.argmax(mass)
    r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
    r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
    std2 = idx[r2] - idx[r1]

    m2 = safe_mode(scan[scan > 0])

    scan_right = scan[scan > 0]
    scan_right = np.clip((scan_right - m2) / std2 + m2, 0, None)
    scan[scan > 0] = scan_right

    mass, idx = np.histogram(scan_right, bins=100)
    i = np.argmax(mass[1:])
    r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
    r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
    std2 = idx[r2] - idx[r1]

    if verbose: print("After mode normalization: Mode2: {:f}, std2: {:f}".format(m2, std2))

    return scan

# Linear, use relative distance of mode2 and mode1 to scale
def mode_norm2(scan, pad_value, verbose=False):
    center = 4

    scan[np.logical_and(scan != pad_value, scan < -1200)] = pad_value
    scan[np.logical_and(scan != pad_value, scan > 600)] = pad_value

    t = -400
    first_stack = scan[np.logical_and(scan != pad_value, scan < t)]
    second_stack = scan[np.logical_and(scan != pad_value, scan > t // 2)]

    mass, idx = np.histogram(first_stack, bins=100)
    i = np.argmax(mass)
    assert i != 0
    r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
    r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
    m1 = safe_mode(first_stack)
    std1 = idx[r2] - idx[r1]
    if verbose: print("Mode1: {:f}, std1: {:f}".format(m1, std1))

    mass, idx = np.histogram(second_stack, bins=100)
    i = np.argmax(mass)
    assert i != 0
    r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
    r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
    m2 = safe_mode(second_stack)
    std2 = idx[r2] - idx[r1]
    if verbose: print("Mode2: {:f}, std2: {:f}".format(m2, std2))

    shift = (m1 + m2) / 2
    scale = center * 2 / (m2 - m1)
    if verbose: print("Shift: {:f}, scale: {:f}".format(shift, scale))

    scan = scan.astype(np.float)
    scan[scan == pad_value] = shift
    # scan[scan > 282.35] = shift ## Remove Bone
    scan = scale * (scan - shift)

    if verbose:
        mass, idx = np.histogram(scan[scan < 0], bins=100)
        i = np.argmax(mass)
        r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
        r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
        std1 = idx[r2] - idx[r1]

        m1 = safe_mode(scan[scan < 0])
        # std1 = np.std(scan[np.logical_and(scan > m1 - 2, scan < np.min(m1 + 2, 0))])
        print("After mode normalization: Mode1: {:f}, std1: {:f}".format(m1, std1))

        mass, idx = np.histogram(scan[scan > 0], bins=100)
        i = np.argmax(mass)
        r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
        r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
        std2 = idx[r2] - idx[r1]

        m2 = safe_mode(scan[scan > 0])
        # std2 = np.std(scan[np.logical_and(scan > m2 - 2, scan < np.min(m2 + 2, 0))])
        print("After mode normalization: Mode2: {:f}, std2: {:f}".format(m2, std2))

    return scan


# Linear, use std of the first stack to scale
def mode_norm(scan, pad_value, verbose=False):

    scan[np.logical_and(scan != pad_value, scan < -1200)] = pad_value
    scan[np.logical_and(scan != pad_value, scan > 600)] = pad_value

    t = -400
    first_stack = scan[np.logical_and(scan != pad_value, scan < t)]
    second_stack = scan[np.logical_and(scan != pad_value, scan > t // 2)]

    mass, idx = np.histogram(first_stack, bins=100)
    i = np.argmax(mass)
    assert i != 0
    r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
    r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
    m1 = safe_mode(first_stack)
    std1 = idx[r2] - idx[r1]
    if verbose: print("Mode1: {:f}, std1: {:f}".format(m1, std1))

    mass, idx = np.histogram(second_stack, bins=100)
    i = np.argmax(mass)
    assert i != 0
    r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
    r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
    m2 = safe_mode(second_stack)
    std2 = idx[r2] - idx[r1]
    if verbose: print("Mode2: {:f}, std2: {:f}".format(m2, std2))

    shift = (m1 + m2) / 2
    scale = 1 / std1
    if verbose: print("Shift: {:f}, scale: {:f}".format(shift, scale))

    scan = scan.astype(np.float)
    scan[scan == pad_value] = shift
    # scan[scan > 282.35] = shift  ## Remove Bone
    scan = scale * (scan - shift)

    if verbose:
        mass, idx = np.histogram(scan[scan < 0], bins=100)
        i = np.argmax(mass)
        r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
        r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
        std1 = idx[r2] - idx[r1]

        m1 = safe_mode(scan[scan < 0])
        # std1 = np.std(scan[np.logical_and(scan > m1 - 2, scan < np.min(m1 + 2, 0))])
        print("After mode normalization: Mode1: {:f}, std1: {:f}".format(m1, std1))

        mass, idx = np.histogram(scan[scan > 0], bins=100)
        i = np.argmax(mass)
        r1 = np.argmin(np.abs(mass[:i] - mass[i] // 2))
        r2 = i + np.argmin(np.abs(mass[i:] - mass[i] // 2))
        std2 = idx[r2] - idx[r1]

        m2 = safe_mode(scan[scan > 0])
        # std2 = np.std(scan[np.logical_and(scan > m2 - 2, scan < np.min(m2 + 2, 0))])
        print("After mode normalization: Mode2: {:f}, std2: {:f}".format(m2, std2))

    return scan

def check_statistic():
    data_dir = "./Methodist_incidental/data_Ben/masked"

    info_path = os.path.join(data_dir, "CTinfo.npz")
    infos = np.load(info_path, allow_pickle=True)["info"].tolist()

    m_ls, t_ls = [], []

    for info in tqdm(infos):
        file_name = info["imagePath"]
        data = np.load(file_name.replace(".npz", "_clean.npz"), allow_pickle=True)
        image = data["image"][0]

        m = safe_mode(image[image < 0])
        t = threshold_otsu(image)

        m_ls.append(m)
        t_ls.append(t)

    plt.hist(m_ls, bins=100)
    plt.title("mode distribution")
    plt.show()

    plt.hist(t_ls, bins=100)
    plt.title("threshold distribution")
    plt.show()

def check_max_luna(pad_value=-3000):
    data_ls = []
    for folder in os.listdir(data_dir):
        for f in os.listdir(os.path.join(data_dir, folder)):
            if f.endswith('_clean.npy'):
                data_ls.append(os.path.join(data_dir, folder, f))
    min_ls, max_ls = [], []
    for data_path in data_ls:
        image = np.load(data_path, allow_pickle=True)
        if len(image) == 1 and len(image.shape) == 4:
            image = image[0]
        min_ls.append(image[image != pad_value].min())
        max_ls.append(image[image != pad_value].max())

    fig, axes = plt.subplots(1, 2, figsize=(6.4 * 2, 4.8))
    axes[0].hist(min_ls, bins=min(100, len(min_ls)))
    axes[0].set_title("min histogram")
    axes[1].hist(max_ls, bins=min(100, len(min_ls)))
    axes[1].set_title("max histogram")
    plt.show()

def mode_norm_single_luna(data_path):
    name = "/".join(data_path.rstrip("_clean.npy").rsplit("/", 2)[1:])
    save_path = os.path.join(save_dir, name+'_label.npy')
    if os.path.exists(save_path):
        return

    image = np.load(data_path, allow_pickle=True)
    if len(image) == 1 and len(image.shape) == 4:
        image = image[0]

    image_norm = mode_func(image, pad_value)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(os.path.join(save_dir, name+'_clean.npy'), image_norm[np.newaxis, ...])
    copyfile(os.path.join(data_dir, name+"_extendbox.npy"), os.path.join(save_dir, name+"_extendbox.npy"))
    copyfile(os.path.join(data_dir, name+"_mask.npy"), os.path.join(save_dir, name+"_mask.npy"))
    copyfile(os.path.join(data_dir, name+"_label.npy"), os.path.join(save_dir, name+"_label.npy"))
    print(save_path)

def mode_normalization_luna():

    data_ls = []
    for folder in os.listdir(data_dir):
        for f in os.listdir(os.path.join(data_dir, folder)):
            if f.endswith('_clean.npy'):
                data_ls.append(os.path.join(data_dir, folder, f))
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)

    for data_path in data_ls:
        mode_norm_single_luna(data_path)

    # pool = Pool(10)
    # _ = pool.map(mode_norm_single_luna, data_ls)

    # ## Single process with for loop
    # for data_path in data_ls:
    #     name = "/".join(data_path.rstrip("_clean.npy").rsplit("/", 2)[1:])
    #     save_path = os.path.join(save_dir, name+'_clean.npy')
    #     if os.path.exists(save_path):
    #         continue
    #
    #     image = np.load(data_path, allow_pickle=True)
    #     if len(image) == 1 and len(image.shape) == 4:
    #         image = image[0]
    #
    #     image = mode_func(image, pad_value)
    #
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     np.save(os.path.join(save_dir, name+'_clean.npy'), image[np.newaxis, ...])
    #     copyfile(os.path.join(data_dir, name+"_extendbox.npy"), os.path.join(save_dir, name+"_extendbox.npy"))
    #     copyfile(os.path.join(data_dir, name+"_mask.npy"), os.path.join(save_dir, name+"_mask.npy"))
    #     copyfile(os.path.join(data_dir, name+"_label.npy"), os.path.join(save_dir, name+"_label.npy"))
    #     print(save_path)


def mode_norm_single_methodist(data_path):
    name = "/".join(data_path.rstrip("_clean.npz").rsplit("/", 2)[1:])
    save_path = os.path.join(save_dir, name+'_label.npz')
    if os.path.exists(save_path):
        return

    image = np.load(data_path, allow_pickle=True)["image"]
    if len(image) == 1 and len(image.shape) == 4:
        image = image[0]

    image_norm = mode_func(image, pad_value)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, name+'_clean.npz'), image=image_norm[np.newaxis, ...])
    copyfile(os.path.join(data_dir, name+"_extendbox.npz"), os.path.join(save_dir, name+"_extendbox.npz"))
    copyfile(os.path.join(data_dir, name+"_mask.npz"), os.path.join(save_dir, name+"_mask.npz"))
    copyfile(os.path.join(data_dir, name+"_label.npz"), os.path.join(save_dir, name+"_label.npz"))
    print(save_path)

def mode_normalization_methodist():
    os.makedirs(save_dir, exist_ok=True)
    pos_label_file = "pos_labels_norm.csv"

    data_ls = []
    for folder in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, folder)):
            for f in os.listdir(os.path.join(data_dir, folder)):
                if f.endswith('_clean.npz'):
                    data_ls.append(os.path.join(data_dir, folder, f))
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)

    pool = Pool(10)
    _ = pool.map(mode_norm_single_methodist, data_ls)
    pool.close()
    pool.join()

    if os.path.exists(os.path.join(data_dir, pos_label_file)):
        copyfile(os.path.join(data_dir, pos_label_file), os.path.join(save_dir, pos_label_file))


    # # label_name = "./Methodist_incidental/data_Ben/maskCropDebug/pos_labels_norm.csv"
    # label_name = os.path.join(data_dir, pos_label_file)
    # pos_df = pd.read_csv(label_name, dtype={"date": str})
    #
    #
    # info_path = os.path.join(data_dir, "CTinfo.npz")
    # infos = np.load(info_path, allow_pickle=True)["info"].tolist()
    # infos_new = []
    #
    # for info in tqdm(infos):
    #
    #     # file_name = "./Methodist_incidental/data_Ben/masked_cropped_rawIntensity/Lung_patient002-025065111/025065111-20110314.npz"
    #     file_name = info["imagePath"]
    #     data = np.load(file_name.replace(".npz", "_clean.npz"), allow_pickle=True)
    #     image = data["image"][0]
    #
    #     pstr, dstr = info["pstr"], info["date"]
    #     existId = (pos_df["patient"] == pstr) & (pos_df["date"] == dstr)
    #     pos = pos_df[existId]
    #     temp = pos[["x", "y", "z", "d"]].values
    #     pos = temp[:, [2, 1, 0, 3]]
    #
    #     extendbox = np.load(file_name.replace(".npz", "_extendbox.npz"))["extendbox"]
    #     pos_new = np.copy(pos).T
    #     pos_new[:3] = pos_new[:3] - np.expand_dims(extendbox[:, 0], 1)
    #     pos = pos_new[:4].T
    #
    #     name = "/".join(file_name.rsplit(".", 1)[0].rsplit("/", 2)[1:])
    #     os.makedirs(os.path.dirname(os.path.join(save_dir, name+"_histNoNorm.png")), exist_ok=True)
    #
    #     # m = safe_mode(image[image != 0])
    #     m = safe_mode(image[image < -400])
    #     # t = threshold_otsu(image)
    #     t = -400
    #
    #     std = np.std(image[image < t])
    #     image_norm = (image - m) / std
    #
    #     # No normalization
    #     plt.figure()
    #     plt.hist(image.reshape(-1), bins=100)
    #     plt.axvline(m, color="r", label="2_nd mode")
    #     plt.axvline(t, color="g", label="threshold")
    #     plt.title("histogram no normalization")
    #     plt.legend()
    #     plt.savefig(os.path.join(save_dir, name+"_histNoNorm.png"), bbox_inches="tight")
    #     plt.close()
    #
    #     for p in pos:
    #         save_path = os.path.join(save_dir, name+"_noNorm")
    #         plot_bbox(save_path, image, p, title="no norm", show=show)
    #
    #
    #     # Thresholding with Otsu's threshold
    #     plt.figure()
    #     plt.hist(image[image < t].reshape(-1), bins=100)
    #     plt.title("histogram first stack")
    #     plt.savefig(os.path.join(save_dir, name+"_histFirstStack.png"), bbox_inches="tight")
    #     plt.close()
    #
    #     for p in pos:
    #         save_path = os.path.join(save_dir, name + "_firstStack")
    #         plot_bbox(save_path, image.clip(None, t), p, title="clip at Otsu's threshold {:d}".format(int(t)), show=show)
    #
    #
    #     # Thresholding with fixed threshold -1200 and 600
    #     plt.figure()
    #     plt.hist(image[np.all([image > -1200, image < 600], axis=0)].reshape(-1), bins=100)
    #     plt.title("histogram clip at -1200, 600")
    #     plt.savefig(os.path.join(save_dir, name+"_histFix.png"), bbox_inches="tight")
    #     plt.close()
    #
    #     for p in pos:
    #         save_path = os.path.join(save_dir, name + "_fix")
    #         plot_bbox(save_path, image.clip(-1200, 600), p, title="clip at -1200, 600", show=show)
    #
    #
    #     # Normalizing with second mode
    #     plt.figure()
    #     plt.hist(image_norm.reshape(-1), bins=100)
    #     plt.title("histogram mode normalization")
    #     plt.savefig(os.path.join(save_dir, name+"_histModeNorm.png"), bbox_inches="tight")
    #     plt.close()
    #
    #     for p in pos:
    #         save_path = os.path.join(save_dir, name + "_modeNorm")
    #         plot_bbox(save_path, image_norm, p, title="mode normalization", show=show)
    #
    #
    #     info["imagePath"] = os.path.join(save_dir, name+'.npz')
    #     np.savez_compressed(os.path.join(save_dir, name+'_clean.npz'), image=image_norm[np.newaxis, ...], info=info)
    #     copyfile(os.path.join(data_dir, name+"_extendbox.npz"), os.path.join(save_dir, name+"_extendbox.npz"))
    #     copyfile(os.path.join(data_dir, name+"_mask.npz"), os.path.join(save_dir, name+"_mask.npz"))
    #
    #     infos_new.append(info)
    #
    # info_new_path = os.path.join(save_dir, "CTinfo.npz")
    # np.savez_compressed(info_new_path, info=infos_new)
    # print("Save all scan infos to {:s}".format(info_new_path))
    #
    # if os.path.exists(os.path.join(data_dir, pos_label_file)):
    #     copyfile(os.path.join(data_dir, pos_label_file), os.path.join(save_dir, pos_label_file))

if __name__ == '__main__':
    mode_func = mode_norm3
    pad_value = -3000
    save_dir = "./LUNA16/modeNorm3_files"
    data_dir = "./LUNA16/masked_files"
    mode_normalization_luna()


    # check_statistic()


    # mode_func = mode_norm
    # pad_value = -3000
    # save_dir = "./Methodist_incidental/data_Ben/modeNorm"
    # data_dir = "./Methodist_incidental/data_Ben/masked"
    # mode_normalization_methodist()

    # check_max_luna()