from argparse import ArgumentParser
from prepare import lumTrans
from show_results import add_bbox
from statistics import mode
from skimage.filters import threshold_otsu
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def load_mhd(data_path, label_path=None, load_label=False):
    '''
    param:
        data_path: endswith .mhd
    return:
         sliceim: npArray, shape == (slices, height, weight)
         label: npArray, shape == (n, 4) -> (z, y, x, d)
    '''
    from prepare import load_itk_image, resample, worldToVoxelCoord
    sliceim, origin, spacing, isflip = load_itk_image(data_path)
    ori_shape = sliceim.shape
    if isflip:
        sliceim = sliceim[:, ::-1, ::-1]
    resolution = np.array([1, 1, 1])
    sliceim, _ = resample(sliceim, spacing, resolution, order=1)
    filename = data_path.split("/")[-1].rstrip(".mhd")
    label = []
    if load_label:
        annos = np.array(pd.read_csv(label_path))
        this_annos = np.copy(annos[annos[:, 0] == (filename)])
        for c in this_annos:
            pos = worldToVoxelCoord(c[1:4][::-1], origin=origin, spacing=spacing)
            if isflip:
                pos[1:] = ori_shape[1:3] - pos[1:]
            label.append(np.concatenate([pos, [c[4] / spacing[1]]]))
        label = np.array(label).T
        if len(label) > 0:
            label[:3] = label[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
            label[3] = label[3] * spacing[1] / resolution[1]
            # label[:3] = label[:3] - np.expand_dims(extendbox[:, 0], 1)
            label = label[:4].T
    label = np.array(label)

    return sliceim, label

def load_npy(data_path, label_path=None, load_label=False):
    '''
    param:
        data_path: endswith .npy
    return:
         sliceim: npArray, shape == (slices, height, weight)
         label: npArray, shape == (n, 4) -> (z, y, x, d)
    '''
    sliceim = np.load(data_path, allow_pickle=True)
    if len(sliceim) == 1 and len(sliceim.shape) == 4:
        sliceim = sliceim[0]
    dirpath = os.path.dirname(data_path)
    filename = data_path.split("/")[-1].rstrip("_clean.npy")
    label = []
    if load_label:
        assert label_path is None
        if label_path is None:
            label = np.load(os.path.join(dirpath, filename + '_label.npy'), allow_pickle=True)
            if np.all(label == 0):
                label = []
    label = np.array(label)
    return sliceim, label

def load_npz(data_path, label_path=None, load_label=False):
    '''
    param:
        data_path: endswith .npy
    return:
         sliceim: npArray, shape == (slices, height, weight)
         label: npArray, shape == (n, 4) -> (z, y, x, d)
    '''
    sliceim = np.load(data_path, allow_pickle=True)["image"]
    if len(sliceim) == 1 and len(sliceim.shape) == 4:
        sliceim = sliceim[0]

    dirpath = os.path.dirname(data_path)
    filename = data_path.split("/")[-1].rstrip("_clean.npz")

    label = []
    if load_label:
        if label_path is None:
            label = np.load(os.path.join(dirpath, filename + '_label.npz'), allow_pickle=True)["label"]
            if np.all(label == 0):
                label = []
        else:
            pos_df = pd.read_csv(label_path)
            pstr, dstr = filename.split("-")
            patient_colname = "patient" if "patient" in pos_df.columns else 'Patient\n Index'
            assert patient_colname in pos_df
            existId = (pos_df[patient_colname] == pstr) & (pos_df["date"] == int(dstr))
            label = pos_df[existId][["z", "y", "x", "d"]].values
    label = np.array(label)
    return sliceim, label

def mode_norm(scan):
    # t = threshold_otsu(scan)
    t = -400
    m = mode(scan[scan < t])
    std = np.std(scan[scan < t])
    print("Mode: {:f}, std: {:f}".format(m, std))
    scan = (scan - m) / std
    return scan

def plot_data(args):

    data_path = args.data_path
    label_path = args.label_path
    nodule_index = args.nodule_index

    load_label = True if nodule_index != -1 else False

    ## load image
    if data_path.endswith(".mhd"):
        scan, label = load_mhd(data_path, label_path, load_label)
    elif data_path.endswith(".npy"):
        scan, label = load_npy(data_path, label_path, load_label)
    else:
        assert data_path.endswith(".npz")
        scan, label = load_npz(data_path, label_path, load_label)
    print("Load data: ", data_path)
    print("Image shape: ", scan.shape)

    ## intensity normalization
    if args.norm == "min_max":
        scan = lumTrans(scan)
    elif args.norm == "mode_norm":
        scan = mode_norm(scan)

    ## load label
    if load_label and nodule_index < len(label):
        z = int(label[nodule_index][0])
        l = label[nodule_index]
        print("All labels: ", label)
    elif args.slice:
        z = args.slice
        l = None
    else:
        z = np.random.randint(0, scan.shape[0])
        l = None

    ## plot data
    fig, axes = plt.subplots(1, 4, figsize=(6.4 * 4, 4.8))
    if l is not None:
        im = add_bbox(axes[0], scan, None, l)
    else:
        im = axes[0].imshow(scan[z], cmap="gray")
    axes[0].set_title("slice {:d}".format(z))
    axes[1].hist(scan.reshape(-1), bins=100)
    axes[1].set_title("histogram")
    fig.colorbar(mappable=im, ax=axes[0])


    ## plot hist
    scan_reshape = scan.reshape(-1)
    bg = mode(scan_reshape)
    scan_reshape = scan_reshape[scan_reshape != bg]
    print("Remove background", bg)
    t = threshold_otsu(scan_reshape)
    first_stack = scan_reshape[scan_reshape < t]
    second_stack = scan_reshape[scan_reshape > t]
    bins1 = min([100, len(set(first_stack))])
    bins2 = min([100, len(set(second_stack))])

    # t = threshold_otsu(scan)
    # first_stack = scan[scan < t].reshape(-1)
    # c = Counter(first_stack)
    # counts = c.most_common(2)
    # if counts[0][1] > 2 * counts[1][1]:
    #     first_stack = first_stack[first_stack != counts[0][0]]
    #     del c[counts[0][0]]
    #     print("Remove background", counts[0])
    # bins1 = min([100, len(c)])
    #
    # second_stack = scan[scan > t].reshape(-1)
    # c = Counter(second_stack)
    # counts = c.most_common(2)
    # if counts[0][1] > 2 * counts[1][1]:
    #     second_stack = second_stack[second_stack != counts[0][0]]
    #     del c[counts[0][0]]
    #     print("remove background", counts[0])
    # bins2 = min([100, len(c)])

    axes[2].hist(first_stack, bins=bins1)
    axes[2].set_title("histogram first stack")
    axes[3].hist(second_stack, bins=bins2)
    axes[3].set_title("histogram second stack")

    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, help="path for the data",
                        default="./LUNA16/raw_files/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd")
    parser.add_argument('-a', '--label_path', type=str, help="path for the label", default=None)
    parser.add_argument('-ni', '--nodule_index', type=int, help="nodule index; not loading any nodule if is -1", default=-1)
    parser.add_argument('-z', '--slice', type=int, default=None, help="the slice index")
    parser.add_argument('-n', '--norm', type=str, choices=["min_max", "mode_norm"], default=None, help="options: [min_max, mode_norm]")
    args = parser.parse_args()

    plot_data(args)
