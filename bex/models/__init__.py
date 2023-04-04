import os
import copy
from .generator import Generator
from .classifier import ResNet, MLP
from .encoder import Encoder
from .configs import default_configs
from ..datasets import get_data_path_or_download


def get_model(model, path, download):

    path = os.path.join(path, "models")
    os.makedirs(path, exist_ok=True)
    model = model.lower()
    configs = copy.deepcopy(default_configs)

    configs[model]["weights"] = get_data_path_or_download(configs[model]["weights"], path, download)

    if "resnet" in model:
        return ResNet(configs[model])

    if model == "encoder":
        return Encoder(configs["encoder"])
    if model == "generator":
        return Generator(configs["generator"])

    raise ValueError("Model %s not found" % model)
