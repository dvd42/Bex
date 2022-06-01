import torch
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
from . import biggan, resnet18


def get_backbone(exp_dict, labelset=None):
    # nclasses = exp_dict["num_classes"]
    backbone_name = exp_dict["backbone"]["name"].lower()
    if backbone_name == "biggan_encoder":
        backbone = biggan.OracleEncoder(exp_dict, labelset)
    elif backbone_name == "biggan_decoder":
        backbone = biggan.Generator(exp_dict, labelset)
    elif backbone_name == "resnet18":
        backbone = resnet18.ResNet18(exp_dict["dataset"]["num_classes"])
    else:
        raise ValueError

    return backbone