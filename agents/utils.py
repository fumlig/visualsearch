import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


def init_weights(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer

def one_hot(x, n):
    return F.one_hot(x.long(), num_classes=n).float()

def normalize_image(image):
    return image/255.0

def channels_first(image):
    assert image.ndim == 4
    return image.permute(0, 3, 1, 2)


def preprocess_image(image):
    return normalize_image(channels_first(image))