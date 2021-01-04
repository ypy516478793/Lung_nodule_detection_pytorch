from skimage.color import gray2rgb
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import scipy.ndimage
import numpy as np
import sys
import os

def getFreeId():
    import pynvml

    pynvml.nvmlInit()
    def getFreeRatio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5*(float(use.gpu+float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(deviceCount):
        if getFreeRatio(i)>70:
            available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus+str(g)+','
    gpus = gpus[:-1]
    return gpus

def setgpu(gpuinput):
    freeids = getFreeId()
    if gpuinput=="all":
        gpus = freeids
    else:
        gpus = gpuinput.replace(" ", "")
        gpu_in_use = [g for g in gpus.split(",") if g not in freeids]
        if len(gpu_in_use) > 0:
            raise ValueError("gpu{:s} is being used! Availabel gpus: gpu {:s}".format(",".join(gpu_in_use), freeids))
    print("using gpu "+gpus)
    os.environ["CUDA_VISIBLE_DEVICES"]=gpus
    return len(gpus.split(","))

def stack_nodule(images, label, prob=None, rows=5, cols=5, show_every=2, patchType="Circle"):
    '''
    Stack slices centered at nodule, return figure
    :param images: slices of CT scan, shape == (nz, h, w)
    :param label: position and the diameter of the nodule, (z, y, x, d)
    :param prob: probability
    :param rows: rows
    :param cols: cols
    :param show_every: show interval
    :param patchType: Circle or Rectangle
    :return: figure
    '''
    fig,ax = plt.subplots(rows,cols,figsize=[9, 9])
    num_show = rows*cols
    z, y, x, d = label
    try:
        nz, h, w = images.shape
    except ValueError:
        print("stop here!")
    start_with = int(z - num_show // 2 * show_every)
    edge_color = "g" if prob is None else "r"
    for i in range(num_show):
        ind = start_with + i*show_every
        ax[int(i/cols),int(i % cols)].set_title('slice %d' % ind)
        if ind < 0 or ind >= nz:
            ax[int(i / cols), int(i % cols)].imshow(np.zeros_like(images[0]), cmap='gray')
        else:
            ax[int(i/cols),int(i % cols)].imshow(images[ind],cmap='gray')
        if patchType == "Circle":
            r = np.sqrt(np.max([0, d * d / 4 - (z - ind) * (z - ind)]))
            rect = patches.Circle((x, y), r, linewidth=1, edgecolor=edge_color, facecolor='none')
        else:
            assert patchType == "Rectangle", "patchType should be either 'Cicle' or 'Rectangle'!"
            rect = patches.Rectangle((x - d / 2, y - d / 2), d, d, linewidth=1, edgecolor=edge_color, facecolor='none')
        ax[int(i/cols),int(i % cols)].add_patch(rect)
        ax[int(i/cols),int(i % cols)].axis('off')
    prob_str = "" if prob is None else "Conf: {:.2f}, ".format(prob)
    st = fig.suptitle("Target: {0:}, {1:s}image shape: {2:}".format(label.round(2), prob_str, images.shape), fontsize="x-large")
    st.set_y(0.95)
    return fig


def stack_nodule_image(images, label, num_show=25, show_every=2):
    """
    Stack slices centered at nodule, manually draw bbox, return image
    :param images: slices of CT scan, shape == (nz, h, w)
    :param label: position and the diameter of the nodule, (z, y, x, d)
    :param num_show: number of slices to show
    :param show_every: show interval
    :return: stacked images centered at nodule
    """
    if isinstance(images[0, 0, 0].item(), int):
        green_color = (0, 128, 0)
    else:
        assert isinstance(images[0, 0, 0].item(), float), "Unknown dtype of images: {}".format(images.dtype)
        green_color = (0, 0.5, 0)

    z, y, x, d = label
    nz, h, w = images.shape
    start_with = int(z - num_show // 2 * show_every)
    assert start_with >= 0, "Need to reduce the 'show_every'! " \
                           "start_with: {:d}, z:{:f}, num_show: {:d}, show_every {:d}".format(
                            start_with, z, num_show, show_every)
    stacks = []
    for i in range(num_show):
        idx = start_with + i * show_every
        rgb_image = gray2rgb(images[idx])
        if z - d / 2 <= idx and z + d / 2 > idx:
            bbox = max(0, y - d / 2), \
                   max(0, x - d / 2), \
                   min(y + d / 2, h), \
                   min(x + d / 2, w)
            rgb_image_bbox = draw_box(rgb_image, bbox, green_color)
            stacks.append(rgb_image_bbox)
    stacks = np.array(stacks)
    return stacks

def draw_box(image, box, color):
    """
    Draw 3-pixel width bounding boxes on the given image array.
    :param color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = np.array(box).astype(np.int)
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image

def get_lr(epoch, args):
    if epoch <= args.epochs * 1 / 3:  # 0.5:
        lr = args.lr
    elif epoch <= args.epochs * 2 / 3:  # 0.8:
        lr = 0.1 * args.lr
    elif epoch <= args.epochs * 0.8:
        lr = 0.05 * args.lr
    else:
        lr = 0.01 * args.lr
    return lr

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def get_logger(logpath, displaying=True, saving=True, debug=False):
    import logging
    logger = logging.getLogger()
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        # info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        # console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def invert_pos(label, thickness, spacing, new_spacing=[1, 1, 1]):
    """
    label: [z, y, x, d]
    """
    spacing = map(float, ([thickness] + list(spacing)))
    spacing = np.array(list(spacing))
    resize_factor = new_spacing / spacing
    label[:3] = np.round(label[:3] * resize_factor)
    label[3] = label[3] * resize_factor[1]
    return label

def invert_image(image, thickness, spacing, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = map(float, ([thickness] + list(spacing)))
    spacing = np.array(list(spacing))
    resize_factor = new_spacing / spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    spacing = new_spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, spacing

def resample_image(image, thickness, spacing, new_spacing=[1, 1, 1]):
    # Determine current pixel spacing
    spacing = map(float, ([thickness] + list(spacing)))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing