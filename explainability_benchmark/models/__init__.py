import os
from .generator import Generator
from .classifier import ResNet, MLP
from .encoder import Encoder
from .configs import default_configs


def get_model(model, path):

    model = model.lower()
    default_configs["encoder"]["weights"] = os.path.join(path, "encoder.pth")


    if default_configs[model]["weights"] is not None:
        default_configs[model]["weights"] = os.path.join(path, default_configs[model]["weights"])

    if "resnet" in model:
        return ResNet(default_configs[model])
    if "mlp" in model:
        return MLP(default_configs[model])

    if model == "encoder":
        return Encoder(default_configs["encoder"])
    if model == "generator":
        # default_configs["generator"]["weights"] = os.path.join(path, "generator.pth")
        return Generator(default_configs["generator"])

    raise ValueError("Model %s not found" % model)
