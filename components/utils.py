import multiprocessing as mp
from torch import nn

imagenet_stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

def parallel(func, arr, num_processes=mp.cpu_count()):
    "Apply some function to an array of values in parallel"
    with mp.Pool(num_processes) as p:
        return p.map(func, arr)

def denorm(im, mean=imagenet_stats[0], std=imagenet_stats[1], image=True):
    "Denormalize an image"
    if isinstance(im, torch.Tensor):
        im = im.detach().clone().cpu().squeeze()
    mean, std = torch.tensor(mean), torch.tensor(std)

    im *= std[..., None, None]
    im += mean[..., None, None]
    im *= 255
    im = im.permute(1, 2, 0).clamp(0,255).numpy()

    im = im.round().astype('uint8')
    if not image: return im
    return Image.fromarray(im)

def norm(im, input_range=(0,255), mean=imagenet_stats[0], std=imagenet_stats[1], unsqueeze=True, grad=True):
    "Normalize an image and set requires_grad"
    if isinstance(im, Image.Image):
        im = torch.tensor(np.asarray(im)).permute(2,0,1).float()
    elif isinstance(im, np.ndarray):
        im = torch.tensor(im).float()
        size = im.size()
        assert len(size)==3 or len(size)==4, "Image has wrong number of dimensions."
        assert size[0]==3 or size[0]==1, "Image has invalid channel number. Should be 1 or 3."

    mean, std = torch.tensor(mean, device=im.device), torch.tensor(std, device=im.device)
    im = im + input_range[0]
    im = im / input_range[1]
    im = im - mean[..., None, None]
    im = im / std[..., None, None]
    if unsqueeze: im.unsqueeze_(0)
    if grad: im.requires_grad_(True)
    return im

def count_params(model):
    return sum(p.numel() for p in model.parameters())
