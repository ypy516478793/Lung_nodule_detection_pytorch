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
import glob
import time
import os


# class IncidentalConfig(object):
#     CROP_LUNG = True
#     MASK_LUNG = True
#     PET_CT = None
#     # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/maskCropDebug"
#     DATA_DIR = "./Methodist_incidental/data_Ben/modeNorm3"
#     # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_Ben/masked_croped_modeNorm"
#     # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/DeepLung-3D_Lung_Nodule_Detection/Methodist_incidental/data_unlabeled/masked_with_crop"
#     # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/"
#     # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/labeled/"
#     # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_king/unlabeled/"
#     # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data/raw_data/unlabeled/"
#     # DATA_DIR = "/home/cougarnet.uh.edu/pyuan2/Projects/Incidental_Lung/data_mamta/processed_data/unlabeled/"
#     # POS_LABEL_FILE = None
#     POS_LABEL_FILE = "pos_labels_norm.csv"
#     # POS_LABEL_FILE = "gt_labels_checklist.xlsx"
#     # POS_LABEL_FILE = "Predicted_labels_checklist_Kim_TC.xlsx"
#     INFO_FILE = "CTinfo.npz"
#     BLACK_LIST = ["001030196-20121205", "005520101-20130316", "009453325-20130820", "034276428-20131212",
#                   "036568905-20150714", "038654273-20160324", "011389806-20160907", "015995871-20160929",
#                   "052393550-20161208", "033204314-20170207", "017478009-20170616", "027456904-20180209",
#                   "041293960-20170227", "000033167-20131213", "022528020-20180525", "025432105-20180730",
#                   "000361956-20180625"]
#
#     ANCHORS = [10.0, 30.0, 60.0]
#     # ANCHORS = [5., 10., 20.]  # [ 10.0, 30.0, 60.]
#     MAX_NODULE_SIZE = 60
#     CHANNEL = 1
#     CROP_SIZE = [96, 96, 96]
#     STRIDE = 4
#     MAX_STRIDE = 16
#     NUM_NEG = 800
#     TH_NEG = 0.02
#     TH_POS_TRAIN = 0.5
#     TH_POS_VAL = 1
#     NUM_HARD = 2
#     BOUND_SIZE = 12
#     RESO = 1
#     SIZE_LIM = 2.5  # 3 #6. #mm
#     SIZE_LIM2 = 10  # 30
#     SIZE_LIM3 = 20  # 40
#     AUG_SCALE = True
#     R_RAND_CROP = 0.3
#     PAD_VALUE = 0   # previous 170
#     AUGTYPE = {"flip": False, "swap": False, "scale": False, "rotate": False, "contrast": False, "bright": False, "sharp": False, "splice": False}
#     # AUGTYPE = {"flip": True, "swap": True, "scale": True, "rotate": True}
#     KFOLD = None
#     KFOLD_SEED = None
#
#     CONF_TH = 4
#     NMS_TH = 0.3
#     DETECT_TH = 0.5
#
#     SIDE_LEN = 144
#     MARGIN = 32
#
#     ORIGIN_SCALE = False
#     SPLIT_SEED = None
#     LIMIT_TRAIN = None
#     SPLIT_ID = None
#
#     def display(self):
#         """Display Configuration values."""
#         print("\nConfigurations:")
#         for a in dir(self):
#             if not a.startswith("__") and not callable(getattr(self, a)):
#                 print("{:30} {}".format(a, getattr(self, a)))
#         print("\n")
Test_fnames = ['patient002_20100910', 'patient002_20110314', 'patient002_20120906', 'patient002_20090310',
               'patient005_20120524', 'patient006_20121023', 'patient011_20120626', 'patient011_20121015',
               'patient012_20121204', 'patient016_20121127', 'patient017_20130102', 'patient018_20121001',
               'patient018_20130110', 'patient021_20120508', 'patient021_20121113', 'patient021_20130212',
               'patient023_20130316', 'patient025_20130226', 'patient032_20130509', 'patient033_20130514',
               'patient034_20121002', 'patient034_20121228', 'patient034_20130423', 'patient035_20160830',
               'patient037_20110516', 'patient037_20111028', 'patient037_20130502', 'patient037_20130510',
               'patient040_20130722', 'patient041_20130319', 'patient041_20130614', 'patient041_20130723',
               'patient045_20130718', 'patient046_20130603', 'patient046_20130826', 'patient047_20130821',
               'patient049_20130820', 'patient050_20130820', 'patient051_20130905', 'patient052_20130516',
               'patient053_20130912', 'patient055_20130925']


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
        self.config = config
        self.subset = subset
        self.data_dir = config.DATA_DIR
        self.blacklist = config.BLACK_LIST
        self.augtype = config.AUGTYPE
        self.stride = config.STRIDE
        self.pad_value = config.PAD_VALUE
        self.r_rand = config.R_RAND_CROP
        self.pet_ct = config.PET_CT

        self.pos_df = None if config.POS_LABEL_FILE is None else \
            pd.read_csv(os.path.join(config.DATA_DIR, config.POS_LABEL_FILE), dtype={"MRN": str, "date": str})
        self.filepaths = self.get_filepaths()
        self.screen()
        self.crop = Crop(config)
        self.split_comber = SplitComb(config.SIDE_LEN, config.MAX_STRIDE, config.STRIDE,
                                      config.MARGIN, config.PAD_VALUE)
        self.label_mapping = LabelMapping(config, subset)
        self.kfold = KFold(n_splits=config.KFOLD, random_state=config.KFOLD_SEED) if config.KFOLD else None
        self.load_subset(subset, random_state=config.SPLIT_SEED, limit_train_size=config.LIMIT_TRAIN,
                         kfold=self.kfold, splitId=config.SPLIT_ID, fixTest=config.FIX_TEST)

        # self.imageInfo = np.load(os.path.join(data_dir, info_file), allow_pickle=True)["info"]

            # if pos_label_file.endswith(".csv"):
            # else:
                # nskip = 0 if pos_label_file == "gt_labels_checklist.xlsx" else 1
                # self.pos_df = pd.read_excel(os.path.join(data_dir, pos_label_file), skiprows=nskip, dtype={"date": str})

        # self.mask_lung = config.MASK_LUNG

        # self.remove_duplicate()

        # if subset != "inference":
        #     self.__check_labels__()


    # def __check_labels__(self):
    #     for info in tqdm(self.imageInfo):
    #         pstr = info["pstr"]
    #         dstr = info["date"]
    #         existId = (self.pos_df["patient"] == pstr) & (self.pos_df["date"] == dstr)
    #         assert existId.sum() > 0, "no matches, pstr {:}, dstr {:}".format(pstr, dstr)

    def get_filepaths(self):
        return glob.glob(self.data_dir + "/*/*_clean.npz")

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

    # def remove_duplicate(self):
    #     for i, info in enumerate(self.imageInfo):
    #         if info["date"] == "":
    #             info["date"] = info["imagePath"].strip(".npz").split("-")[-1]
    #
    #     identifier_set = ["{:}-{:}".format(info["patientID"], info["date"]) for info in self.imageInfo]
    #     remove_ids = []
    #     from collections import Counter
    #     cnt = Counter(identifier_set)
    #     for k, v in cnt.items():
    #         if k in self.blacklist:
    #             indices = [i for i, x in enumerate(identifier_set) if x == k]
    #             remove_ids = remove_ids + indices
    #         elif v > 1:
    #             indices = [i for i, x in enumerate(identifier_set) if x == k]
    #             remove_ids = remove_ids + indices[:-1]
    #     self.imageInfo = np.delete(self.imageInfo, remove_ids)

    def screen(self):
        '''
        Remove scan in the blacklist.
        '''
        # TODO: remove scan in the blacklist.
        pass

        # num_images = len(self.imageInfo)
        # print("number of CT scans: {:d}".format(num_images))
        # mask = np.ones(num_images, dtype=bool)
        # if self.pet_ct is not None:
        #     for imageId in range(num_images):
        #         info = self.imageInfo[imageId]
        #         if (self.pet_ct and info["PET"] == "N") or (not self.pet_ct and info["PET"] == "Y"):
        #             mask[imageId] = False
        #         # pos = self.load_pos(imageId)
        #         # if len(pos) == 0:
        #         #     mask[imageId] = False
        #     self.imageInfo = self.imageInfo[mask]
        #     print("number of CT scans after screening: {:d}".format(len(self.imageInfo)))

    def load_subset(self, subset, random_state=None, limit_train_size=None, kfold=None, splitId=None, fixTest=False):
        if subset == "inference":
            return ## Nothing needs to be modified in inference mode

        ## Train/val/test split for scans
        if random_state is None:
            random_state = 42


        if kfold is None:
            if fixTest:
                trainValScan, testScan = [], []
                for fpath in self.filepaths:
                    fname = fpath.split("/")[-1].rstrip("_clean.npz").replace("-", "_")
                    if fname in Test_fnames:
                        testScan.append(fpath)
                    else:
                        trainValScan.append(fpath)
                trainScan, valScan = train_test_split(trainValScan, test_size=0.25, random_state=random_state)
            else:
                trainScan, valTestScan = train_test_split(self.filepaths, test_size=0.6, random_state=random_state)
                valScan, testScan = train_test_split(valTestScan, test_size=0.5, random_state=random_state)
        else:
            assert splitId is not None
            if fixTest:
                trainValScan, testScan = [], []
                for fpath in self.filepaths:
                    fname = fpath.split("/")[-1].rstrip("_clean.npz")
                    if fname in Test_fnames:
                        testScan.append(fpath)
                    else:
                        trainValScan.append(fpath)
            else:
                trainValScan, testScan = train_test_split(self.filepaths, test_size=0.2, random_state=random_state)
            kf_indices = [(train_index, val_index) for train_index, val_index in kfold.split(trainValScan)]
            train_index, val_index = kf_indices[splitId]
            trainScan, valScan = trainValScan[train_index], trainValScan[val_index]

        assert subset == "train" or subset == "val" or subset == "test", "Unknown subset!"
        if subset == "train":
            scans = trainScan
            if limit_train_size is not None:
                scans = scans[:int(limit_train_size * len(scans))]
        elif subset == "val":
            scans = valScan
        else:
            scans = testScan
        self.filepaths = scans

        ## Get the file list for current subset
        # start = infos[0]["imagePath"].find("Lung_patient")
        # fileList = [i["imagePath"][start:] for i in infos]
        # if subset != "test":
        #     fileList = [f for f in fileList if (f not in self.blacklist)]
        # self.filenames = [os.path.join(self.data_dir, f) for f in fileList]
        ## Load the label for current subset

        ## Load all nodules
        labels = []
        print("Subset {:s} has {:d} samples.".format(subset, len(self.filepaths)))
        if self.pos_df is None:
            for data_path in self.filepaths:
                dirname = os.path.dirname(data_path)
                filename = data_path.split("/")[-1].rstrip("_clean.npz")
                label = np.load(os.path.join(dirname, filename + '_label.npz'), allow_pickle=True)["label"]
                label = label[label[:, -1] < self.config.MAX_NODULE_SIZE] ## remove nodule larger than specific size
                if np.all(label == 0):
                    label = np.array([])
                labels.append(label)
        else:
            for data_path in self.filepaths:
                filename = data_path.split("/")[-1].rstrip("_clean.npz")
                pstr, dstr = filename.split("-")
                patient_colname = "patient" if "patient" in self.pos_df.columns else 'Patient\n Index'
                assert patient_colname in self.pos_df
                existId = (self.pos_df[patient_colname] == pstr) & (self.pos_df["date"] == int(dstr))
                label = self.pos_df[existId][["z", "y", "x", "d"]].values
                label = label[label[:, -1] < self.config.MAX_NODULE_SIZE] ## remove nodule larger than specific size
                labels.append(label)

        self.sample_bboxes = labels
                # info = self.search_info(filePath)
                # assert info != -1, "No matched info for {:s}!!".format(filePath)
                # l = self.load_pos(info)
                #
                # # print("")
                # if self.config.CROP_LUNG:
                #     extendbox = np.load(info["imagePath"].replace(".npz", "_extendbox.npz"))["extendbox"]
                #
                #     if len(l) == 0:
                #         l = np.array([[0, 0, 0, 0]])
                #     else:
                #         ll = np.copy(l).T
                #         ll[:3] = ll[:3] - np.expand_dims(extendbox[:, 0], 1)
                #         l = ll[:4].T
                #
                # if np.all(l == 0):
                #     l = np.array([])
                # labels.append(l)

        ## Duplicate samples based on the nodule size
        if self.subset != "test":
            self.bboxes = []
            for i, l in enumerate(labels):
                for t in l:
                    if t[3] > self.config.SIZE_LIM:
                        self.bboxes.append([np.concatenate([[i], t])])
                    if t[3] > self.config.SIZE_LIM2:
                        self.bboxes += [[np.concatenate([[i], t])]] * 2
                    if t[3] > self.config.SIZE_LIM3:
                        self.bboxes += [[np.concatenate([[i], t])]] * 4
            if len(self.bboxes) > 0:
                self.bboxes = np.concatenate(self.bboxes, axis=0)

    # def search_info(self, path):
    #     for info in self.imageInfo:
    #         # if info["imagePath"].strip(".") in path:
    #         # if info["imagePath"] == path:
    #         if path.strip("./") in info["imagePath"]:
    #             return info
    #     return -1

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

    ## plot histogram of the nodule diameters
    def plot_nodule_hist(self):
        all_pos = [self.load_pos(i) for i in self.imageInfo]
        all_pos = np.concatenate(all_pos)
        plt.hist(all_pos[:, -1], bins=100)
        plt.title("Histogram of the nodule diameters")
        plt.xlabel("Diameter (ps)")
        plt.ylabel("Count")
        plt.savefig("Histogram_d.png", bbox_inches="tight", dpi=200)

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
                temp = np.load(self.filepaths[idx], allow_pickle=True)
                info = temp["info"]
                imgs = temp["image"]
            except:
                temp = np.load(self.filepaths[idx].replace(".npz", "_clean.npz"), allow_pickle=True)
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
                filepath = self.filepaths[int(bbox[0])]

                imgs = np.load(filepath, allow_pickle=True)["image"]

                # try:
                #     imgs = np.load(filename, allow_pickle=True)["image"]
                # except:
                #     imgs = np.load(filename.replace(".npz", "_clean.npz"), allow_pickle=True)["image"]
                #     imgs = imgs.squeeze(0)
                # if not self.mask_lung:
                #     imgs = lumTrans(imgs) # fixme: Need to decide when to do normalization
                #
                # pass
                #
                # imgs = imgs[np.newaxis, :]
                bboxes = self.sample_bboxes[int(bbox[0])]
                isScale = self.augtype["scale"] and (self.subset == "train")
                # isScale = False
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale, isRandom)
                if sample.shape != (1, 96, 96, 96):
                    print("")
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
            # print sample.shape, target.shape, bboxes.shape
            label = self.label_mapping(sample.shape[1:], target, bboxes, filepath)
            sample = (sample.astype(np.float32) - 128) / 128
            # if filename in self.kagglenames and self.subset=="train":
            #    label[label==-1]=0
            if sample.shape != (1, 96, 96, 96):
                print("")
            if sample.shape[2] != 96:
                print("")
            return torch.from_numpy(sample), torch.from_numpy(label), coord, target
        else:
            # try:
            #     imgs = np.load(self.filepaths[idx], allow_pickle=True)["image"]
            # except:
            #     imgs = np.load(self.filepaths[idx].replace(".npz", "_clean.npz"), allow_pickle=True)["image"]
            #     imgs = imgs.squeeze(0)


            # if not self.mask_lung:
            #     imgs = lumTrans(imgs)
            # imgs = imgs[np.newaxis, :]
            imgs = np.load(self.filepaths[idx], allow_pickle=True)["image"]
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
            return len(self.filepaths)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from show_results import plot_bbox
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.utils import make_grid

    writer = SummaryWriter(os.path.join("Visualize", "MethodistFull"))

    test = False
    inference = False
    config = IncidentalConfig()
    config.SPLIT_SEED = 128

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
        pos = temp[:, [2, 1, 0, 3]]

        extendbox = np.load(info["imagePath"].replace(".npz", "_extendbox.npz"))["extendbox"]
        ll = np.copy(pos).T
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

    from detector_ben.utils import stack_nodule
    fig = stack_nodule(sample[1, 0], target[1].numpy())
    plt.show()
    plt.close(fig)


    print()
