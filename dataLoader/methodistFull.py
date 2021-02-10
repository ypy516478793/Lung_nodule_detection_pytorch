from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import rotate
from dataLoader.dataBase import LabelMapping, Crop, collate
from dataLoader.splitCombine import SplitComb
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from skimage import morphology
from skimage import measure
import pandas as pd
import numpy as np
import torch
import time
import os

class IncidentalConfig(object):
    MASK_LUNG = True
    PET_CT = False

    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/labeled/"
    if MASK_LUNG:
        DATA_DIR = "/data/pyuan2/Methodist_incidental/data_kim/masked_first/"
    else:
        DATA_DIR = "/data/pyuan2/Methodist_incidental/data_kim/labeled/"

    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/unlabeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/raw_data/unlabeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_mamta/processed_data/unlabeled/"
    INFO_FILE = "CTinfo.npz"
    POS_LABEL_FILE = "pos_labels.csv"
    # POS_LABEL_FILE = "gt_labels_checklist.xlsx"
    # POS_LABEL_FILE = "Predicted_labels_checklist_Kim_TC.xlsx"
    # POS_LABEL_FILE = None
    BLACK_LIST = []

    ANCHORS = [10.0, 30.0, 60.0]
    MAX_NODULE_SIZE = 60
    # ANCHORS = [5., 10., 20.]  # [ 10.0, 30.0, 60.]
    CHANNEL = 1
    CROP_SIZE = [96, 96, 96]
    STRIDE = 4
    MAX_STRIDE = 16
    NUM_NEG = 800
    TH_NEG = 0.02
    TH_POS_TRAIN = 0.5
    TH_POS_VAL = 1
    NUM_HARD = 2
    BOUND_SIZE = 12
    RESO = 1
    SIZE_LIM = 2.5  # 3 #6. #mm
    SIZE_LIM2 = 10  # 30
    SIZE_LIM3 = 20  # 40
    AUG_SCALE = True
    R_RAND_CROP = 0.3
    PAD_VALUE = 0   # previous 170
    AUGTYPE = {"flip": True, "swap": False, "scale": True, "rotate": False}

    CONF_TH = 4
    NMS_TH = 0.3
    DETECT_TH = 0.5

    SIDE_LEN = 144
    MARGIN = 32

    ORIGIN_SCALE = False

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


def resample_pos(label, thickness, spacing, new_spacing=[1, 1, 1]):
    spacing = map(float, ([thickness] + list(spacing)))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    resize_factor = resize_factor[::-1]
    label[:3] = np.round(label[:3] * resize_factor)
    label[3] = label[3] * resize_factor[1]
    return label

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype("uint8")
    return newimg

def mask_scan(images):
    masked_images = []
    for img in images:
        masked_images.append(make_lungmask(img))
    masked_images = np.stack(masked_images)
    return masked_images
    # plt.imshow(images[10])
    # print("Images{:d} shape: ".format(imageId), images.shape)

def make_lungmask(img, display=False):
    raw_img = np.copy(img)
    row_size = img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img)
    if std == 0:
        return np.zeros_like(img)

    img = img - mean
    img = img / std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the
    # underflow and overflow on the pixel spectrum
    img[img == max] = mean
    img[img == min] = mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))

    labels = measure.label(dilation)  # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 10 and B[
            2] < col_size / 10 * 9:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0  # mask = np.zeros([row_size, col_size], dtype=np.int8)

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))  # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask * img, cmap='gray')
        ax[2, 1].axis('off')

        plt.show()
    return mask * img

def augment(sample, target, bboxes, coord, ifflip=True, ifrotate=True, ifswap=True):
    #                     angle1 = np.random.rand()*180
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

class MethodistFull(Dataset):
    def __init__(self, config, subset="train"):
        assert (subset == "train" or subset == "val" or subset == "test" or subset == "inference")
        self.config = config
        self.subset = subset
        self.data_dir = data_dir = config.DATA_DIR
        self.blacklist = config.BLACK_LIST
        info_file = config.INFO_FILE
        pos_label_file = config.POS_LABEL_FILE
        self.augtype = config.AUGTYPE
        self.stride = config.STRIDE
        self.pad_value = config.PAD_VALUE
        self.r_rand = config.R_RAND_CROP
        self.split_comber = SplitComb(config.SIDE_LEN, config.MAX_STRIDE, config.STRIDE,
                                      config.MARGIN, config.PAD_VALUE)
        self.imageInfo = np.load(os.path.join(data_dir, info_file), allow_pickle=True)["info"]
        if pos_label_file is not None:
            self.pos_df = pd.read_csv(os.path.join(data_dir, pos_label_file), dtype={"date": str})
        self.crop = Crop(config)
        self.mask_lung = config.MASK_LUNG
        self.pet_ct = config.PET_CT
        self.screen()
        self.label_mapping = LabelMapping(config, subset)
        self.load_subset(subset)

    def screen(self):
        '''
        Remove nodule size >= 60
        '''
        num_images = len(self.imageInfo)
        mask = np.ones(num_images, dtype=bool)
        if self.pet_ct is not None:
            for imageId in range(num_images):
                info = self.imageInfo[imageId]
                if (self.pet_ct and info["PET"] == "Y") or (not self.pet_ct and info["PET"] == "N"):
                    mask[imageId] = False
                # pos = self.load_pos(imageId)
                # if len(pos) == 0:
                #     mask[imageId] = False
            self.imageInfo = self.imageInfo[mask]

    def load_subset(self, subset):
        ## train/val/test split
        if subset == "inference":
            infos = self.imageInfo
        else:
            ## train/val/test split
            trainInfo, valInfo = train_test_split(self.imageInfo, test_size=0.6, random_state=42)
            valInfo, testInfo = train_test_split(valInfo, test_size=0.5, random_state=42)

            assert subset == "train" or subset == "val" or subset == "test", "Unknown subset!"
            if subset == "train":
                infos = trainInfo
            elif subset == "val":
                infos = valInfo
            else:
                # infos = testInfo
                infos = trainInfo

        ## Get the file list for current subset
        start = infos[0]["imagePath"].find("Lung_patient")
        fileList = [i["imagePath"][start:] for i in infos]
        if subset != "test":
            fileList = [f for f in fileList if (f not in self.blacklist)]
        self.filenames = [os.path.join(self.data_dir, f) for f in fileList]

        # self.filenames = [i["imagePath"] for i in self.imageInfo]
        # self.filenames = [os.path.join(data_dir, "%s_clean.npy" % idx) for idx in idcs]
        # print self.filenames
        # self.kagglenames = [f for f in self.filenames]  # if len(f.split("/")[-1].split("_")[0])>20]
        # self.lunanames = [f for f in self.filenames if len(f.split("/")[-1].split("_")[0])<20]

        ## Load the label for current subset
        labels = []
        print("Subset {:s} has {:d} samples.".format(subset, len(fileList)))
        if "pos_df" in dir(self):
            for filePath in self.filenames:
                info = self.search_info(filePath)
                assert info != -1, "No matched info!!"
                l = self.load_pos(info)
                # print data_dir, idx
                # l = np.load(data_dir+idx+"_label.npy",allow_pickle="True")
                # print l, os.path.join(data_dir, "%s_label.npy" %idx)
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

    def search_info(self, path):
        for info in self.imageInfo:
            if info["imagePath"] == path:
                return info
        return -1

    def load_pos(self, imgInfo):
        thickness, spacing = imgInfo["sliceThickness"], imgInfo["pixelSpacing"]
        pstr = imgInfo["pstr"]
        dstr = imgInfo["date"]
        existId = (self.pos_df["patient"] == pstr) & (self.pos_df["date"] == dstr)
        pos = self.pos_df[existId][["x", "y", "z", "d"]].values
        pos = np.array([resample_pos(p, thickness, spacing) for p in pos])
        pos = pos[:, [2, 1, 0, 3]]
        pos = pos[pos[:, -1] < self.config.MAX_NODULE_SIZE]

        return pos

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
            temp = np.load(self.filenames[idx], allow_pickle=True)
            imgs = temp["image"]
            if not self.mask_lung:
                imgs = lumTrans(imgs)
            imgs = imgs[np.newaxis, :]
            info = temp["info"]
            ori_imgs = np.copy(imgs)
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], "constant",
                          constant_values=self.pad_value)

            xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] / self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[2] / self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[3] / self.stride), indexing="ij")
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
                imgs = np.load(filename, allow_pickle=True)["image"]
                if not self.mask_lung:
                    imgs = lumTrans(imgs)
                imgs = imgs[np.newaxis, :]
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
                imgs = np.load(filename, allow_pickle=True)["image"]
                if self.mask_lung:
                    imgs = lumTrans(imgs)
                imgs = imgs[np.newaxis, :]
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
            imgs = np.load(self.filenames[idx], allow_pickle=True)["image"]
            if not self.mask_lung:
                imgs = lumTrans(imgs)
            imgs = imgs[np.newaxis, :]
            ori_imgs = np.copy(imgs)
            bboxes = self.sample_bboxes[idx]
            nz, nh, nw = imgs.shape[1:]
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], "constant",
                          constant_values=self.pad_value)

            xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] / self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[2] / self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[3] / self.stride), indexing="ij")
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

    writer = SummaryWriter(os.path.join("Visualize", "MethodistFull"))

    config = IncidentalConfig()
    dataset = MethodistFull(config, subset="train")

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
    # sample, label, coord, target = next(iterator)
    from tqdm import tqdm
    i = 0
    for sample, label, coord, target in tqdm(iterator):
        print(i)
        i += 1
    # sample, label, coord, target = next(iterator)

    from detector_ben.utils import stack_nodule
    fig = stack_nodule(sample[1, 0], target[1].numpy())
    plt.show()
    plt.close(fig)


    # img_grid = make_grid(x1[:, :, cube_size // 2])

    print()



# def label2target(label, i):
#     a = label[i]
#     ids = np.where(a[..., 0] == 1)
#     ids = np.concatenate(ids)
#     print("ids is: ", ids)
#     config = IncidentalConfig()
#     stride = config.STRIDE
#     pos = ids[:3]
#     # coord[0, :, *pos]
#     offset = (stride - 1) / 2
#     oh = np.arange(offset, offset + stride * (config.CROP_SIZE[0] - 1) + 1, stride)
#     c = oh[pos]
#     l = label[i, ids[0], ids[1], ids[2], ids[3]]
#     l[-1] = np.exp(l[-1])
#     ll = (l[1:] * 20).numpy() + np.array(c.tolist() + [0,])
#     plot_bbox(None, sample[0, 0], None, label=ll[[2, 1, 0, 3]])


# ## plot histogram of the nodule diameters
# all_pos = [self.load_pos(i) for i in self.imageInfo]
# all_pos = np.concatenate(all_pos)
# plt.hist(all_pos[:, -1], bins=100)
# plt.title("Histogram of the nodule diameters")
# plt.xlabel("Diameter (ps)")
# plt.ylabel("Count")
# plt.savefig("Histogram_d.png", bbox_inches="tight", dpi=200)