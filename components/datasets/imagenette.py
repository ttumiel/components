from pathlib import Path
from torchvision import datasets
from components.transforms.basic import get_transforms_torchvision
from components.datasets.download import download, unpack_data

imagenette_mean = [0.4616, 0.4538, 0.4254]
imagenette_std = [0.2681, 0.2643, 0.2865]

def imagenette(path, imsize, download=False, **kwargs):
    "Utility function to download Imagenette dataset"
    path = Path(path)
    if download:
        path.mkdir(exist_ok=True)
        url = imagenette_url(imsize)
        download(path/Path(url).name, url)
        unpack_data(path/url.name)
    train_tfms, val_tfms = get_transforms_torchvision(imsize, imagenette_mean, imagenette_std, **kwargs)
    train_ds = datasets.ImageFolder(path/'train', transform=train_tfms)
    valid_ds = datasets.ImageFolder(path/'val', transform=val_tfms)
    return train_ds, valid_ds

def imagenette_url(imsize):
    if imsize <=160:
        return 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'
    elif imsize < 320:
        return 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'
    else:
        return 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'
