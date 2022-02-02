from scipy.ndimage.interpolation import zoom
from nilearn.image import resample_img
import nibabel as nib
import numpy as np



def load_nii(data_path):
    """ return image and spacing """
    hdr = nib.load(data_path)
    image = hdr.get_fdata()
    spacing = np.abs(hdr.affine.diagonal()[:3])
    # spacing = np.array(hdr.header.get_zooms())
    image = image.transpose(2, 1, 0)
    spacing = spacing[[2, 1, 0]]
    image = image[::-1, ::-1, ::-1]
    return image, spacing

def load_nii_resample(data_path):
    """ resample to new spacing [1, 1, 1], return new image and old spacing """
    hdr = nib.load(data_path)
    spacing = np.abs(hdr.affine.diagonal()[:3])
    hdr.set_data_dtype(np.float32)
    # spacing = np.array(hdr.header.get_zooms())
    new_hdr = resample_img(hdr, target_affine=np.eye(3))
    new_image = new_hdr.get_fdata()
    new_image = new_image.transpose(2, 1, 0)
    spacing = spacing[[2, 1, 0]]
    new_image = new_image[::-1, ::-1, ::-1]
    return new_image, spacing

def resample_image(image, spacing, new_spacing=np.array([1, 1, 1])):
    """ resample use skimage; spacing (npArray): [thickness, H, W] """
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = zoom(image, real_resize_factor)

    return image, new_spacing