import skimage.io
from PIL import Image
import numpy as np
import torch

def open_tiff_image(path, level=2):
    """
    Opens a .tiff image. These images can have multiple
    'levels', like sizes of the image.
    """
    return skimage.io.MultiImage(str(path))[level]

def pad_to_square(im, min_size=256, fill_color=(0,0,0)):
    "Pad a PIL Image to a square."
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def crop_to_mask(im, mask):
    "Crops an image using a boolean mask of the same shape."
    im = np.asarray(im)
    c = np.nonzero(mask)
    top_left = np.min(c, axis=1)
    bottom_right = np.max(c, axis=1)
    out = im[top_left[0]:bottom_right[0],
                top_left[1]:bottom_right[1]]
    return out

def show_im(data):
    if isinstance(data, Image.Image):
        return data
    if isinstance(data, np.ndarray):
        return Image.fromarray(data)
    if isinstance(data, torch.Tensor):
        return Image.fromarray(data.cpu().detach().numpy())
