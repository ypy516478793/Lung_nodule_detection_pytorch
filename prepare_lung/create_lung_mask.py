from scipy.ndimage import binary_fill_holes
from skimage import morphology, measure
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def lung_mask_3D(images, max_ind=None, format="nifti"):
    """ Create lung mask based on 3D connectivity """
    if format == "npz":
        # Transpose the image from [z, y, x] to [x, y, z], and reverse each axis
        images = images.transpose(2, 1, 0)
        images = images[::-1, ::-1, ::-1]

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

    lung_found = False
    threshold = 0.8
    while not lung_found and threshold > 0.5:
        mask0 = np.copy(mask)
        # Assume 40% threshold for body cross section is neck position
        # ind=find(head_to_toe_profile>.4*max(head_to_toe_profile));
        head_to_toe_profile = np.sum(mask0, axis=(0, 1))
        # Assume 40% threshold for body cross section is neck position
        max_area = max(head_to_toe_profile[head_to_toe_profile != mask0.shape[0] * mask0.shape[1]])
        ind = np.where(head_to_toe_profile > threshold * max_area)[0]  # change from 0.4 to 0.6, to 0.7
        # Smart masking of neck and up
        mask0[1: -1, 1: -1, : min(ind) + 1] = 1
        if max_ind is not None:
            mask0[1: -1, 1: -1, max_ind:] = 1
        else:
            mask0[1: -1, 1: -1, max(ind):] = 1
        CC = measure.label(1 - mask0, connectivity=2)
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
                mask0[CC == idx[i]] = -1  # back_ground
            else:
                locs = np.where(CC == idx[i])
                if (
                        locs[0].min() < x / 2 and
                        locs[0].max() > x / 2 and
                        locs[1].min() < y / 2 and
                        locs[1].max() > y / 2
                ):
                    mask0[CC == idx[i]] = 2  # lung
                    find_lung = True # local lung found
                    lung_found = True # global lung found
                    mask = mask0
                else:
                    mask0[CC == idx[i]] = 1  # other area
            if i >= 5:
                break # no lung is found
        threshold = threshold - 0.1 # change another threshold
    if not lung_found:
        print("no lung detected")
        return None

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

    mask = (mask == 2).astype(np.int8)

    if format == "npz":
        # Transpose and reverse back the mask
        mask = mask.transpose(2, 1, 0)
        mask = mask[::-1, ::-1, ::-1]

    return mask

def lung_mask_2D(images, format="npz"):
    """ Create lung mask slice by slice (2D) """
    if format == "nifti":
        # Transpose the image from [x, y, z] to [z, y, x], and reverse each axis
        images = images.transpose(2, 1, 0)
        images = images[::-1, ::-1, ::-1]

    masks = []
    for img in images:
        mask = lung_mask_slice(img)
        masks.append(mask)
    masks = np.stack(masks)

    if format == "nifti":
        # Transpose and reverse back the mask
        masks = masks.transpose(2, 1, 0)
        masks = masks[::-1, ::-1, ::-1]
    return masks

def lung_mask_slice(img, display=False):
    """ Create lung mask for one slice (2D) """
    row_size = img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img)
    if std == 0:
        return np.zeros_like(img), np.zeros_like(img, dtype=np.int8)

    img = img - mean
    img = img / std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    # middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    middle = img[0:col_size, 0:row_size]
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

    eroded = morphology.erosion(thresh_img, np.ones([5, 5]))
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
    mask = morphology.dilation(mask, np.ones([12, 12]))  # one last dilation

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
    return mask

