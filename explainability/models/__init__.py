import torch
import torchvision.models as models

import time
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from .autoencoder import Autoencoder
from .tcvae import TCVAE


def get_model(exp_dict, **kwargs):
    if exp_dict["model"] == "autoencoder":
        return Autoencoder(exp_dict, **kwargs)
    elif exp_dict["model"] == "tcvae":
        return TCVAE(exp_dict, **kwargs)
    else:
        raise ValueError("Model %s not found" %exp_dict["model"])
