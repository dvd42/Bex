from .generator import Generator
from .classifier import ResNet
from .encoder import Encoder
import sys
import os
# sys.path.append("..")
from .configs import default_configs


def get_model(model, path):

    model = model.lower()
    default_configs["classifier"]["weights"] = os.path.join(path, "classifier_font_char.pth")
    default_configs["encoder"]["weights"] = os.path.join(path, "encoder.pth")
    default_configs["generator"]["weights"] = os.path.join(path, "generator.pth")

    if model == "resnet":
        return ResNet(default_configs["classifier"])
    if model == "encoder":
        return Encoder(default_configs["encoder"])
    if model == "generator":
        return Generator(default_configs["generator"])

    raise ValueError("Model %s not found" % model)
