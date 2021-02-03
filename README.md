# Components

A library of ML components for faster experiments, research, blogposts, competitions and fun.


## Models

Create various neural networks from different components. See the [`components/models`](/components/models) folder.

- MobileNet v1, v2, v3
- ResNet, XResNet
- SENet
- EfficientNet

## Datasets

For faster loading of various data sources. See the [`components/datasets`](/components/datasets) folder.

## Training

Easily train models using [PyTorch Lightning](https://www.pytorchlightning.ai/) or a vanilla training script. Includes grid search functionality, easy configuration, telemetry and logging. See [`components/lightning`](/components/lightning).

## Examples

See [`nbs`](/nbs) for examples inside Jupyter notebooks.

## Install

```bash
# Clone repo
git clone https://github.com/ttumiel/components && cd components

# Editable install (so you can change the code and iterate)
python -m pip install -e .
```

## Blogposts

- [From MobileNet to EfficientNet](https://ttumiel.github.io/blog/mobilenet-to-efficientnet/)
- [Replacing BatchNorm](https://ttumiel.github.io/blog/replacing-bn/)

