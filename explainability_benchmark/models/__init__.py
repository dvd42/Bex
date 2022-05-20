import os
import copy
from .generator import Generator
from .classifier import ResNet, MLP
from .encoder import Encoder
from .configs import default_configs


def get_model(model, path):

    model = model.lower()
    configs = copy.deepcopy(default_configs)
    configs["encoder"]["weights"] = os.path.join(path, default_configs["encoder"]["weights"])

    if default_configs[model]["weights"] is not None:
        configs[model]["weights"] = os.path.join(path, configs[model]["weights"])

    if "resnet" in model:
        return ResNet(configs[model])
    if "mlp" in model:
        return MLP(configs[model])

    if model == "encoder":
        return Encoder(configs["encoder"])
    if model == "generator":
        return Generator(configs["generator"])

    raise ValueError("Model %s not found" % model)
