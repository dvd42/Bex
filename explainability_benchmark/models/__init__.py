import os
from .generator import Generator
from .classifier import ResNet, MLP
from .encoder import Encoder
from .configs import default_configs


def get_model(model, path):

    model = model.lower()

    default_configs["encoder"]["weights"] = os.path.join(path, default_configs["encoder"]["weights"])
    default_configs["generator"]["weights"] = os.path.join(path, default_configs["generator"]["weights"])

    if model == "resnet":
        if default_configs["resnet"]["weights"] is not None:
            default_configs["resnet"]["weights"] = os.path.join(path, default_configs["resnet"]["weights"])
        return ResNet(default_configs["resnet"])
    if model == "mlp":
        if default_configs["mlp"]["weights"] is not None:
            default_configs["mlp"]["weights"] = os.path.join(path, default_configs["mlp"]["weights"])
        return MLP(default_configs["mlp"])

    if model == "encoder":
        return Encoder(default_configs["encoder"])
    if model == "generator":
        return Generator(default_configs["generator"])

    raise ValueError("Model %s not found" % model)
