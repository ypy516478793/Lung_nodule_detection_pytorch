from skimage.filters import threshold_otsu
from show_results import plot_bbox
from statistics import mode
from shutil import copyfile
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def check_statistic():
    data_dir = "./Methodist_incidental/data_Ben/masked_cropped_rawIntensity"

    info_path = os.path.join(data_dir, "CTinfo.npz")
    infos = np.load(info_path, allow_pickle=True)["info"].tolist()

    m_ls, t_ls = [], []

    for info in tqdm(infos):
        file_name = info["imagePath"]
        data = np.load(file_name.replace(".npz", "_clean.npz"), allow_pickle=True)
        image = data["image"][0]

        m = mode(image[image < 0])
        t = threshold_otsu(image)

        m_ls.append(m)
        t_ls.append(t)

    plt.hist(m_ls, bins=100)
    plt.title("mode distribution")
    plt.show()

    plt.hist(t_ls, bins=100)
    plt.title("threshold distribution")
    plt.show()


def main():
    show = False

    save_dir = "./Methodist_incidental/data_Ben/modeNorm"
    data_dir = "./Methodist_incidental/data_Ben/masked"
    os.makedirs(save_dir, exist_ok=True)
    pos_label_file = "pos_labels_norm.csv"

    # label_name = "./Methodist_incidental/data_Ben/maskCropDebug/pos_labels_norm.csv"
    label_name = os.path.join(data_dir, pos_label_file)
    pos_df = pd.read_csv(label_name, dtype={"date": str})


    info_path = os.path.join(data_dir, "CTinfo.npz")
    infos = np.load(info_path, allow_pickle=True)["info"].tolist()
    infos_new = []

    for info in tqdm(infos):

        # file_name = "./Methodist_incidental/data_Ben/masked_cropped_rawIntensity/Lung_patient002-025065111/025065111-20110314.npz"
        file_name = info["imagePath"]
        data = np.load(file_name.replace(".npz", "_clean.npz"), allow_pickle=True)
        image = data["image"][0]

        pstr, dstr = info["pstr"], info["date"]
        existId = (pos_df["patient"] == pstr) & (pos_df["date"] == dstr)
        pos = pos_df[existId]
        temp = pos[["x", "y", "z", "d"]].values
        pos = temp[:, [2, 1, 0, 3]]

        extendbox = np.load(file_name.replace(".npz", "_extendbox.npz"))["extendbox"]
        pos_new = np.copy(pos).T
        pos_new[:3] = pos_new[:3] - np.expand_dims(extendbox[:, 0], 1)
        pos = pos_new[:4].T

        name = "/".join(file_name.rsplit(".", 1)[0].rsplit("/", 2)[1:])
        os.makedirs(os.path.dirname(os.path.join(save_dir, name+"_histNoNorm.png")), exist_ok=True)

        # m = mode(image[image != 0])
        m = mode(image[image < -400])
        # t = threshold_otsu(image)
        t = -400

        std = np.std(image[image < t])
        image_norm = (image - m) / std

        # No normalization
        plt.figure()
        plt.hist(image.reshape(-1), bins=100)
        plt.axvline(m, color="r", label="2_nd mode")
        plt.axvline(t, color="g", label="threshold")
        plt.title("histogram no normalization")
        plt.legend()
        plt.savefig(os.path.join(save_dir, name+"_histNoNorm.png"), bbox_inches="tight")
        plt.close()

        for p in pos:
            save_path = os.path.join(save_dir, name+"_noNorm")
            plot_bbox(save_path, image, p, title="no norm", show=show)


        # Thresholding with Otsu's threshold
        plt.figure()
        plt.hist(image[image < t].reshape(-1), bins=100)
        plt.title("histogram first stack")
        plt.savefig(os.path.join(save_dir, name+"_histFirstStack.png"), bbox_inches="tight")
        plt.close()

        for p in pos:
            save_path = os.path.join(save_dir, name + "_firstStack")
            plot_bbox(save_path, image.clip(None, t), p, title="clip at Otsu's threshold {:d}".format(int(t)), show=show)


        # Thresholding with fixed threshold -1200 and 600
        plt.figure()
        plt.hist(image[np.all([image > -1200, image < 600], axis=0)].reshape(-1), bins=100)
        plt.title("histogram clip at -1200, 600")
        plt.savefig(os.path.join(save_dir, name+"_histFix.png"), bbox_inches="tight")
        plt.close()

        for p in pos:
            save_path = os.path.join(save_dir, name + "_fix")
            plot_bbox(save_path, image.clip(-1200, 600), p, title="clip at -1200, 600", show=show)


        # Normalizing with second mode
        plt.figure()
        plt.hist(image_norm.reshape(-1), bins=100)
        plt.title("histogram mode normalization")
        plt.savefig(os.path.join(save_dir, name+"_histModeNorm.png"), bbox_inches="tight")
        plt.close()

        for p in pos:
            save_path = os.path.join(save_dir, name + "_modeNorm")
            plot_bbox(save_path, image_norm, p, title="mode normalization", show=show)


        info["imagePath"] = os.path.join(save_dir, name+'.npz')
        np.savez_compressed(os.path.join(save_dir, name+'_clean.npz'), image=image_norm[np.newaxis, ...], info=info)
        copyfile(os.path.join(data_dir, name+"_extendbox.npz"), os.path.join(save_dir, name+"_extendbox.npz"))
        copyfile(os.path.join(data_dir, name+"_mask.npz"), os.path.join(save_dir, name+"_mask.npz"))

        infos_new.append(info)

    info_new_path = os.path.join(save_dir, "CTinfo.npz")
    np.savez_compressed(info_new_path, info=infos_new)
    print("Save all scan infos to {:s}".format(info_new_path))

    if os.path.exists(os.path.join(data_dir, pos_label_file)):
        copyfile(os.path.join(data_dir, pos_label_file), os.path.join(save_dir, pos_label_file))

if __name__ == '__main__':
    main()
    # check_statistic()