import os
import copy
from .generator import Generator
from .classifier import ResNet, MLP
from .encoder import Encoder
from .configs import default_configs
from ..datasets import get_data_path_or_download


def get_model(model, path):

    path = os.path.join(path, "models")
    os.makedirs(path, exist_ok=True)
    model = model.lower()
    configs = copy.deepcopy(default_configs)

    # configs["encoder"]["weights"] = get_data_path_or_download(configs["encoder"]["weights"], path)
    configs[model]["weights"] = get_data_path_or_download(configs[model]["weights"], path)

    # if default_configs[model]["weights"] is not None:
    #     configs[model]["weights"] = os.path.join(path, configs[model]["weights"])

    if "resnet" in model:
        return ResNet(configs[model])

    if model == "encoder":
        return Encoder(configs["encoder"])
    if model == "generator":
        return Generator(configs["generator"])

    raise ValueError("Model %s not found" % model)
