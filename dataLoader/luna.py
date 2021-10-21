from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import rotate
from dataLoader.dataBase import LabelMapping, Crop, collate
from dataLoader.splitCombine import SplitComb
from dataLoader.lunaConfig import LunaConfig
from torch.utils.data import Dataset
from prepare import lumTrans
import pandas as pd
import numpy as np
import torch
import time
import os


def augment(sample, target, bboxes, coord, ifflip=True, ifrotate=True, ifswap=True):
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand() * 180
            size = np.array(sample.shape[2:4]).astype("float")
            rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)],
                               [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
            newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample, angle1, axes=(2, 3), reshape=False)
                coord = rotate(coord, angle1, axes=(2, 3), reshape=False)
                for box in bboxes:
                    box[1:3] = np.dot(rotmat, box[1:3] - size / 2) + size / 2
            else:
                counter += 1
                if counter == 3:
                    break
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
            coord = np.transpose(coord, np.concatenate([[0], axisorder + 1]))
            target[:3] = target[:3][axisorder]
            bboxes[:, :3] = bboxes[:, :3][:, axisorder]

    if ifflip:
        #         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        coord = np.ascontiguousarray(coord[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        for ax in range(3):
            if flipid[ax] == -1:
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                bboxes[:, ax] = np.array(sample.shape[ax + 1]) - bboxes[:, ax]
    return sample, target, bboxes, coord


class Luna(Dataset):
    def __init__(self, config, subset="train"):

        assert (subset == 'train' or subset == 'val' or subset == 'test' or subset == "inference")
        self.config = config
        self.subset = subset
        self.data_dir = config.DATA_DIR
        self.train_data_dir = config.TRAIN_DATA_DIR
        self.val_data_dir = config.VAL_DATA_DIR
        self.test_data_dir = config.TEST_DATA_DIR
        self.blacklist = config.BLACK_LIST
        self.augtype = config.AUGTYPE
        self.stride = config.STRIDE
        self.pad_value = config.PAD_VALUE
        self.r_rand = config.R_RAND_CROP
        self.split_comber = SplitComb(config.SIDE_LEN, config.MAX_STRIDE, config.STRIDE,
                                      config.MARGIN, config.PAD_VALUE)
        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, subset)
        self.load_subset(subset)

    def load_subset(self, subset):

        fileList = []
        assert subset == "train" or subset == "val" or subset == "test" or subset == "inference", "Unknown subset!"
        if subset == "train":
            data_dir = self.train_data_dir
        elif subset == "val":
            data_dir = self.val_data_dir
        else:
            data_dir = self.test_data_dir

        for folder in data_dir:
            for f in os.listdir(os.path.join(self.data_dir, folder)):
                if f.endswith('_clean.npy') and f.rstrip("_clean.npy") not in self.blacklist:
                    fileList.append(folder + '/' + f.rstrip("_clean.npy"))

        self.filenames = [os.path.join(self.data_dir, "{:s}_clean.npy".format(idx)) for idx in fileList]
        # self.filenames = [os.path.join(self.data_dir, f) for f in fileList]

        labels = []

        print(len(fileList))
        for idx in fileList:
            label_path = os.path.join(self.data_dir, "{:s}_label.npy".format(idx))
            l = np.load(label_path, allow_pickle=True).astype(np.float32)
            if np.all(l == 0):
                l = np.array([])
            labels.append(l)

            ## Duplicate samples based on the nodule size
        self.sample_bboxes = labels
        if self.subset != "test":
            self.bboxes = []
            for i, l in enumerate(labels):
                if len(l) > 0:
                    for t in l:
                        if t[3] > self.config.SIZE_LIM:
                            self.bboxes.append([np.concatenate([[i], t])])
                        if t[3] > self.config.SIZE_LIM2:
                            self.bboxes += [[np.concatenate([[i], t])]] * 2
                        if t[3] > self.config.SIZE_LIM3:
                            self.bboxes += [[np.concatenate([[i], t])]] * 4
            if len(self.bboxes) > 0:
                self.bboxes = np.concatenate(self.bboxes, axis=0)

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time

        isRandomImg = False
        if self.subset == "train" or self.subset == "val":
            if idx >= len(self.bboxes):
                isRandom = True
                idx = idx % len(self.bboxes)
                # isRandomImg = np.random.randint(2)
            else:
                isRandom = False
        else:
            isRandom = False

        if self.subset == "inference":
            imgs = np.load(self.filenames[idx], allow_pickle=True)
            # imgs = temp["image"]
            info = np.array([])
            # imgs = lumTrans(imgs)
            ori_imgs = np.copy(imgs)
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], "constant",
                          constant_values=self.pad_value)

            xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] // self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[2] // self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[3] // self.stride), indexing="ij")
            coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype("float32")
            imgs, nzhw = self.split_comber.split(imgs)
            coord2, nzhw2 = self.split_comber.split(coord,
                                                    side_len=self.split_comber.side_len / self.stride,
                                                    max_stride=self.split_comber.max_stride / self.stride,
                                                    margin=self.split_comber.margin / self.stride)
            assert np.all(nzhw == nzhw2)
            imgs = (imgs.astype(np.float32) - 128) / 128
            return torch.from_numpy(imgs), None, torch.from_numpy(coord2), np.array(nzhw), ori_imgs, info

        if self.subset != "test":
            if not isRandomImg:
                bbox = self.bboxes[idx]
                filename = self.filenames[int(bbox[0])]
                imgs = np.load(filename, allow_pickle=True)
                # imgs = lumTrans(imgs)
                bboxes = self.sample_bboxes[int(bbox[0])]
                # isScale = self.augtype["scale"] and (self.subset == "train")
                isScale = False
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale, isRandom)
                if self.subset == "train" and not isRandom:
                    sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                                                            ifflip=self.augtype["flip"],
                                                            ifrotate=self.augtype["rotate"],
                                                            ifswap=self.augtype["swap"])
            else:
                randimid = np.random.randint(len(self.kagglenames))
                filename = self.kagglenames[randimid]
                imgs = np.load(filename, allow_pickle=True)
                # imgs = lumTrans(imgs)
                bboxes = self.sample_bboxes[randimid]
                isScale = self.augtype["scale"] and (self.subset == "train")
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes, isScale=False, isRand=True)
            # print sample.shape, target.shape, bboxes.shape
            label = self.label_mapping(sample.shape[1:], target, bboxes, filename)
            sample = (sample.astype(np.float32) - 128) / 128
            # if filename in self.kagglenames and self.subset=="train":
            #    label[label==-1]=0
            return torch.from_numpy(sample), torch.from_numpy(label), coord, target
        else:
            imgs = np.load(self.filenames[idx], allow_pickle=True)
            # imgs = lumTrans(imgs)
            ori_imgs = np.copy(imgs)
            bboxes = self.sample_bboxes[idx]
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], "constant",
                          constant_values=self.pad_value)

            xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] // self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[2] // self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[3] // self.stride), indexing="ij")
            coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype("float32")
            imgs, nzhw = self.split_comber.split(imgs)
            coord2, nzhw2 = self.split_comber.split(coord,
                                                    side_len=self.split_comber.side_len / self.stride,
                                                    max_stride=self.split_comber.max_stride / self.stride,
                                                    margin=self.split_comber.margin / self.stride)
            assert np.all(nzhw == nzhw2)
            imgs = (imgs.astype(np.float32) - 128) / 128
            return torch.from_numpy(imgs), bboxes, torch.from_numpy(coord2), np.array(nzhw), ori_imgs

    def __len__(self):
        if self.subset == "train":
            return int(len(self.bboxes) / (1 - self.r_rand))
            # return len(self.bboxes)
        elif self.subset == "val":
            return len(self.bboxes)
        else:
            return len(self.filenames)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from show_results import plot_bbox
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.utils import make_grid

    writer = SummaryWriter(os.path.join("Visualize", "lunaRaw"))

    config = LunaConfig()
    dataset = Luna(config, subset="train")

    # inference_loader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     collate_fn=collate,
    #     pin_memory=False)
    #
    # iterator = iter(inference_loader)
    # cropped_sample, target, coord, nzhw, sample, info = next(iterator)

    # test_loader = DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     collate_fn=collate,
    #     pin_memory=False)
    #
    # iterator = iter(test_loader)
    # cropped_sample, target, coord, nzhw, sample = next(iterator)

    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True)

    iterator = iter(train_loader)
    sample, label, coord, target = next(iterator)
    from detector_ben.utils import stack_nodule

    fig = stack_nodule(sample[0][0], target[0][0])
    plt.show()
    plt.close(fig)

    # img_grid = make_grid(x1[:, :, cube_size // 2])

    print()