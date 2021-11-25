import torch
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
from . import biggan, resnet18


def get_backbone(exp_dict, labelset):
    # nclasses = exp_dict["num_classes"]
    backbone_name = exp_dict["backbone"]["name"].lower()
    if backbone_name == "biggan":
        backbone = biggan.Autoencoder(exp_dict, labelset)
    elif backbone_name == "resnet":
        backbone = resnet18.ResNet18()
    elif backbone_name == "vae":
        backbone = biggan.VAE(exp_dict, labelset)
    else:
        raise ValueError

    return backbone
