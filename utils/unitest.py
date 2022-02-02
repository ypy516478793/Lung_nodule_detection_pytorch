from tqdm import tqdm
import numpy as np
import glob
import os


data_dir = "./Methodist_incidental/data_Ben/preprocessed_data_v1"
filepaths = glob.glob(data_dir+"/*/*_clean.npz")
print("Check {:d} scans".format(len(filepaths)))
for filepath in tqdm(filepaths):
    imgs = np.load(filepath, allow_pickle=True)["image"]
    if not np.all(imgs.shape[1:] > np.array([96, 96, 96])):
        raise ValueError("Need to check {:s}".format(filepath))

print("All images are good in shape!")