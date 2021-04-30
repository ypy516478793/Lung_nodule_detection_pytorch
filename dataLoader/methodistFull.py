from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from scipy.ndimage.interpolation import rotate
from dataLoader.dataBase import LabelMapping, Crop, collate
from dataLoader.splitCombine import SplitComb
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from skimage import morphology
from skimage import measure
from tqdm import tqdm
import imgaug.augmenters as iaa
import pandas as pd
import numpy as np
import torch
import time
import os

exclude = ["001030196-20121205", "005520101-20130316", "009453325-20130820", "034276428-20131212",
           "036568905-20150714", "038654273-20160324", "011389806-20160907", "015995871-20160929",
           "052393550-20161208", "033204314-20170207", "017478009-20170616", "027456904-20180209",
           "041293960-20170227", "000033167-20131213", "022528020-20180525", "025432105-20180730",
           "000361956-20180625"]

class IncidentalConfig(object):
    CROP_LUNG = False
    MASK_LUNG = False
    PET_CT = False
    # ROOT_DIR = "/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_kim/"
    # ROOT_DIR = "/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_Ben/"
    # ROOT_DIR = "/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_unlabeled/"
    # ROOT_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben"
    DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/maskCropDebug"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/labeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/unlabeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/raw_data/unlabeled/"
    # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_mamta/processed_data/unlabeled/"
    INFO_FILE = "CTinfo.npz"
    POS_LABEL_FILE = "pos_labels_norm.csv"
    # POS_LABEL_FILE = "pos_labels_norm.csv"
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
    AUGTYPE = {"flip": False, "swap": False, "scale": False, "rotate": False, "contrast": False, "bright": False, "sharp": False, "splice": False}
    # AUGTYPE = {"flip": True, "swap": True, "scale": True, "rotate": True}
    KFOLD = None
    KFOLD_SEED = None

    CONF_TH = 4
    NMS_TH = 0.3
    DETECT_TH = 0.5

    SIDE_LEN = 144
    MARGIN = 32

    ORIGIN_SCALE = False
    SPLIT_SEED = None
    LIMIT_TRAIN = None
    SPLIT_ID = None

    # def set_data_dir(self):
    #     if self.MASK_LUNG:
    #         if self.CROP_LUNG:
    #             self.DATA_DIR = os.path.join(self.ROOT_DIR, "masked_with_crop")
    #         else:
    #             self.DATA_DIR = os.path.join(self.ROOT_DIR, "masked_first")
    #     else:
    #         self.DATA_DIR = os.path.join(self.ROOT_DIR, "labeled")

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

def splice(sample_bgd, target_bgd, bboxes_bgd, coord_bgd, sample, target):
    r = target[3] / 2
    start = np.array(target[:3] - r).astype(np.int)
    end = np.array(target[:3] + r).astype(np.int)
    sample_bgd[0, start[0]: end[0], start[1]: end[1], start[2]: end[2]] = \
        sample[0, start[0]: end[0], start[1]: end[1], start[2]: end[2]]
    return sample_bgd, target, bboxes_bgd, coord_bgd

def augment(sample, target, bboxes, coord, ifflip=False, ifrotate=False, ifswap=False, ifcontrast=False, ifbright=False, ifsharp=False):
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
    if ifcontrast:
        factor = np.random.rand() * 2
        new_sample = []
        for i in range(sample.shape[1]):
            image_pil = Image.fromarray(sample[0, i])
            enhancer = ImageEnhance.Contrast(image_pil)
            image_enhanced = enhancer.enhance(factor)
            new_sample.append(np.array(image_enhanced))
        sample = np.expand_dims(new_sample, 0)

    if ifbright:
        factor = np.random.rand() * 2
        new_sample = []
        for i in range(sample.shape[1]):
            image_pil = Image.fromarray(sample[0, i])
            enhancer = ImageEnhance.Brightness(image_pil)
            image_enhanced = enhancer.enhance(factor)
            new_sample.append(np.array(image_enhanced))
        sample = np.expand_dims(new_sample, 0)

    if ifsharp:
        factor = np.random.rand() * 2
        new_sample = []
        for i in range(sample.shape[1]):
            image_pil = Image.fromarray(sample[0, i])
            enhancer = ImageEnhance.Sharpness(image_pil)
            image_enhanced = enhancer.enhance(factor)
            new_sample.append(np.array(image_enhanced))
        sample = np.expand_dims(new_sample, 0)

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
            if pos_label_file.endswith(".csv"):
                self.pos_df = pd.read_csv(os.path.join(data_dir, pos_label_file), dtype={"MRN": str, "date": str})
            else:
                nskip = 0 if pos_label_file == "gt_labels_checklist.xlsx" else 1
                self.pos_df = pd.read_excel(os.path.join(data_dir, pos_label_file), skiprows=nskip, dtype={"date": str})
        self.crop = Crop(config)
        self.mask_lung = config.MASK_LUNG
        self.pet_ct = config.PET_CT
        self.remove_duplicate()
        self.screen()
        if subset != "inference":
            self.__check_labels__()
        self.label_mapping = LabelMapping(config, subset)
        self.kfold = KFold(n_splits=config.KFOLD, random_state=config.KFOLD_SEED) if config.KFOLD else None
        self.load_subset(subset, random_state=config.SPLIT_SEED, limit_train_size=config.LIMIT_TRAIN,
                         kfold=self.kfold, splitId=config.SPLIT_ID)

        # ---- plot bbox ---- #
        # info = self.imageInfo[1]
        # pstr = info["pstr"]
        # # s = info["imagePath"].find("Lung_patient")
        # # p = os.path.join("/home/cougarnet.uh.edu/pyuan2/Datasets/Methodist_incidental/data_kim/labeled/",
        # #                  info["imagePath"][s:].replace("\\", "/"))
        # p = info["imagePath"]
        # img = lumTrans(np.load(p)["image"])
        # pos = self.load_pos(info)
        # from show_results import plot_bbox
        # save_dir = "reports"
        # os.makedirs(save_dir, exist_ok=True)
        # plot_bbox(os.path.join(save_dir, "{:s}_newLabel.png".format(pstr)), img, None, pos[0], show=False,
        #           title="{:s}_newLabel".format(pstr))

        # ---- check labels ---- #
        # for info in self.imageInfo:
        #     img = np.load(info["imagePath"], allow_pickle=True)["image"]
        #     img = lumTrans(img)
        #     pos = self.load_pos(info)
        #     for p in pos:
        #         plot_bbox(None, img, p, title=info["pstr"])

        # self.aug_op = "flip_rot"
        # self.augmentor = self.set_augment()

        # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        # self.seq = iaa.Sequential([iaa.Fliplr(0.5),
        #                            iaa.Flipud(0.5),
        #                            sometimes(iaa.Rot90([1, 3])),
        #                            sometimes(iaa.Affine(rotate=(-45, 45))),
        #                            iaa.TranslateY(px=(-40, 40)),
        #                            iaa.TranslateX(px=(-40, 40)),
        #                            ])

    def __check_labels__(self):
        for info in tqdm(self.imageInfo):
            pstr = info["pstr"]
            dstr = info["date"]
            existId = (self.pos_df["patient"] == pstr) & (self.pos_df["date"] == dstr)
            assert existId.sum() > 0, "no matches, pstr {:}, dstr {:}".format(pstr, dstr)

    def set_augment(self):
        if self.aug_op == "flip_rot":
            augmentor = iaa.Sequential([iaa.Fliplr(0.5), iaa.Rot90([0, 3])])
        elif self.aug_op == "gamma_contrast":
            augmentor = iaa.GammaContrast((2.0))
        elif self.aug_op == "shift":
            augmentor = iaa.Sequential([iaa.TranslateY(px=(-40, 40)), iaa.TranslateX(px=(-40, 40))])
        else:
            assert self.aug_op == "all"
            augmentor = iaa.Sequential([iaa.Fliplr(0.5), iaa.Rot90([0, 3]),
                                        iaa.GammaContrast((2.0)),
                                        iaa.TranslateY(px=(-40, 40)), iaa.TranslateX(px=(-40, 40))])

        return augmentor

    def remove_duplicate(self):
        for i, info in enumerate(self.imageInfo):
            if info["date"] == "":
                info["date"] = info["imagePath"].strip(".npz").split("-")[-1]

        identifier_set = ["{:}-{:}".format(info["patientID"], info["date"]) for info in self.imageInfo]
        remove_ids = []
        from collections import Counter
        cnt = Counter(identifier_set)
        for k, v in cnt.items():
            if k in exclude:
                indices = [i for i, x in enumerate(identifier_set) if x == k]
                remove_ids = remove_ids + indices
            elif v > 1:
                indices = [i for i, x in enumerate(identifier_set) if x == k]
                remove_ids = remove_ids + indices[:-1]
        self.imageInfo = np.delete(self.imageInfo, remove_ids)

    def screen(self):
        '''
        Remove nodule size >= 60
        '''
        num_images = len(self.imageInfo)
        print("number of CT scans: {:d}".format(num_images))
        mask = np.ones(num_images, dtype=bool)
        if self.pet_ct is not None:
            for imageId in range(num_images):
                info = self.imageInfo[imageId]
                if (self.pet_ct and info["PET"] == "N") or (not self.pet_ct and info["PET"] == "Y"):
                    mask[imageId] = False
                # pos = self.load_pos(imageId)
                # if len(pos) == 0:
                #     mask[imageId] = False
            self.imageInfo = self.imageInfo[mask]
            print("number of CT scans after screening: {:d}".format(len(self.imageInfo)))

    def load_subset(self, subset, random_state=None, limit_train_size=None, kfold=None, splitId=None):
        ## train/val/test split
        if subset == "inference":
            infos = self.imageInfo
        else:
            ## train/val/test split
            if random_state is None:
                random_state = 42
            if kfold is None:
                trainInfo, valInfo = train_test_split(self.imageInfo, test_size=0.6, random_state=random_state)
                valInfo, testInfo = train_test_split(valInfo, test_size=0.5, random_state=random_state)
            else:
                assert splitId is not None
                trainValInfo, testInfo = train_test_split(self.imageInfo, test_size=0.2, random_state=random_state)
                kf_indices = [(train_index, val_index) for train_index, val_index in kfold.split(trainValInfo)]
                train_index, val_index = kf_indices[splitId]
                trainInfo, valInfo = trainValInfo[train_index], trainValInfo[val_index]


            assert subset == "train" or subset == "val" or subset == "test", "Unknown subset!"
            if subset == "train":
                infos = trainInfo
                if limit_train_size is not None:
                    infos = infos[:int(limit_train_size * len(infos))]
            elif subset == "val":
                infos = valInfo
            else:
                infos = testInfo
                # infos = trainInfo

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
                assert info != -1, "No matched info for {:s}!!".format(filePath)
                l = self.load_pos(info)

                # print("")
                if self.config.CROP_LUNG:
                    extendbox = np.load(info["imagePath"].replace(".npz", "_extendbox.npz"))["extendbox"]

                    if len(l) == 0:
                        l = np.array([[0, 0, 0, 0]])
                    else:
                        ll = np.copy(l).T
                        # label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
                        # label2[3] = label2[3] * spacing[1] / resolution[1]
                        ll[:3] = ll[:3] - np.expand_dims(extendbox[:, 0], 1)
                        l = ll[:4].T

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
        patient_colname = "patient" if "patient" in self.pos_df.columns else 'Patient\n Index'
        assert patient_colname in self.pos_df
        existId = (self.pos_df[patient_colname] == pstr) & (self.pos_df["date"] == dstr)
        pos = self.pos_df[existId][["x", "y", "z", "d"]].values
        # pos[:, 2] = pos[:, 2] - 1  ## BUG FIXED: slice index begins with 1 in EPIC system
        # pos = np.array([resample_pos(p, thickness, spacing) for p in pos])
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
            try:
                temp = np.load(self.filenames[idx], allow_pickle=True)
                info = temp["info"]
                imgs = temp["image"]
            except:
                temp = np.load(self.filenames[idx].replace(".npz", "_clean.npz"), allow_pickle=True)
                info = temp["info"]
                imgs = temp["image"]
                imgs = imgs.squeeze(0)
            if not self.mask_lung:
                imgs = lumTrans(imgs)
            imgs = imgs[np.newaxis, :]
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
                try:
                    imgs = np.load(filename, allow_pickle=True)["image"]
                except:
                    imgs = np.load(filename.replace(".npz", "_clean.npz"), allow_pickle=True)["image"]
                    imgs = imgs.squeeze(0)
                if not self.mask_lung:
                    imgs = lumTrans(imgs)

                pass

                imgs = imgs[np.newaxis, :]
                bboxes = self.sample_bboxes[int(bbox[0])]
                isScale = self.augtype["scale"] and (self.subset == "train")
                # isScale = False
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale, isRandom)
                if self.subset == "train" and not isRandom:
                    sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                                                            ifflip=self.augtype["flip"],
                                                            ifrotate=self.augtype["rotate"],
                                                            ifswap=self.augtype["swap"],
                                                            ifcontrast=self.augtype["contrast"],
                                                            ifbright=self.augtype["bright"],
                                                            ifsharp=self.augtype["sharp"],)
                    if self.augtype["splice"]:
                        sample_bgd, target_bgd, bboxes_bgd, coord_bgd = self.crop(
                            imgs, bbox[1:], bboxes, isScale=False, isRand=True)
                        sample, target, bboxes, coord = splice(sample_bgd, target_bgd, bboxes_bgd, coord_bgd, sample,
                                                               target)
            else:
                randimid = np.random.randint(len(self.kagglenames))
                filename = self.kagglenames[randimid]
                try:
                    imgs = np.load(filename, allow_pickle=True)["image"]
                except:
                    imgs = np.load(filename.replace(".npz", "_clean.npz"), allow_pickle=True)["image"]
                    imgs = imgs.squeeze(0)
                if self.mask_lung:
                    imgs = lumTrans(imgs)

                pass

                imgs = imgs[np.newaxis, :]
                bboxes = self.sample_bboxes[randimid]
                isScale = self.augtype["scale"] and (self.subset == "train")
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes, isScale, isRand=True)
            # print sample.shape, target.shape, bboxes.shape
            label = self.label_mapping(sample.shape[1:], target, bboxes, filename)
            sample = (sample.astype(np.float32) - 128) / 128
            # if filename in self.kagglenames and self.subset=="train":
            #    label[label==-1]=0
            if sample.shape[2] != 96:
                print("")
            return torch.from_numpy(sample), torch.from_numpy(label), coord, target
        else:
            try:
                imgs = np.load(self.filenames[idx], allow_pickle=True)["image"]
            except:
                imgs = np.load(self.filenames[idx].replace(".npz", "_clean.npz"), allow_pickle=True)["image"]
                imgs = imgs.squeeze(0)
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

    test = True
    inference = False
    config = IncidentalConfig()
    # config.set_data_dir()
    config.SPLIT_SEED = 128
    # config.LIMIT_TRAIN = args.limit_train
    # config.AUGTYPE["flip"] = args.flip
    # config.AUGTYPE["swap"] = args.swap
    # config.AUGTYPE["scale"] = args.scale
    # config.AUGTYPE["rotate"] = args.rotate
    # config.AUGTYPE["contrast"] = args.contrast
    # config.AUGTYPE["bright"] = args.bright
    # config.AUGTYPE["sharp"] = args.sharp
    # config.AUGTYPE["splice"] = args.splice
    # config.KFOLD = args.kfold
    # config.SPLIT_ID = args.split_id

    if test:
        subset = "test"
    elif inference:
        subset = "inference"
    else:
        subset = "train"
    dataset = MethodistFull(config, subset=subset)


    from show_results import plot_bbox
    data_dir = config.DATA_DIR
    pos_label_file = config.POS_LABEL_FILE
    pos_df = pd.read_csv(os.path.join(data_dir, pos_label_file), dtype={"date": str})
    for i, info in enumerate(dataset.imageInfo):
        print(i)
        pstr = info["pstr"]
        pid = int(pstr[7:])
        if pid <= 51:
            continue

        filename = info["imagePath"]
        try:
            img = np.load(filename, allow_pickle=True)["image"]
        except:
            img = np.load(filename.replace(".npz", "_clean.npz"), allow_pickle=True)["image"]
            img = img.squeeze(0)

        filename = info["imagePath"]
        pstr = info["pstr"]
        dstr = info["date"]
        thickness = info["sliceThickness"]
        spacing = info["pixelSpacing"]
        existId = (pos_df["patient"] == pstr) & (pos_df["date"] == dstr)
        pos = pos_df[existId]
        temp = pos[["x", "y", "z", "d"]].values
        # temp[:, 2] = temp[:, 2] - 1
        # temp = np.array([resample_pos(p, thickness, spacing) for p in temp])
        pos = temp[:, [2, 1, 0, 3]]

        extendbox = np.load(info["imagePath"].replace(".npz", "_extendbox.npz"))["extendbox"]
        ll = np.copy(pos).T
        # label2[:3] = label2[:3] * np.expand_dims(spacing, 1) / np.expand_dims(resolution, 1)
        # label2[3] = label2[3] * spacing[1] / resolution[1]
        ll[:3] = ll[:3] - np.expand_dims(extendbox[:, 0], 1)
        pos = ll[:4].T

        for p in pos:
            plot_bbox(None, img, p)

    if subset == "inference":
        inference_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
            pin_memory=False)

        iterator = iter(inference_loader)
        cropped_sample, target, coord, nzhw, sample, info = next(iterator)

    if subset == "test":
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
            pin_memory=False)

        iterator = iter(test_loader)
        cropped_sample, target, coord, nzhw, sample = next(iterator)
        i = 0
        for sample, label, coord, nzhw, image in tqdm(iterator):
            print(i)
            i += 1

    if subset == "train":
        train_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            pin_memory=True)

        iterator = iter(train_loader)
        i = 0
        for sample, label, coord, target in tqdm(iterator):
            print(i)
            i += 1
    # sample, label, coord, target = next(iterator)

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