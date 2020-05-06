import skimage.io
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import math

from components.utils import denorm

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

def show_batch(dataset, n=16, denorm=False, figsize=(12,12), random=True):
    "Visualise a batch of a dataset"
    r = math.ceil(math.sqrt(n))
    axes = plt.subplots(r,r,figsize=figsize)[1].flatten()
    for i,ax in enumerate(axes):
        if i<n:
            im,label = dataset[np.random.randint(len(ds))] if random else ds[i]
            if isinstance(im, torch.Tensor):
                im = denorm(im)
            ax.imshow(im)
            ax.set_title(f'{label}')
        ax.set_axis_off()
