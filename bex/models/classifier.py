import torch
from .backbones import get_backbone


class MLP(torch.nn.Module):

    def __init__(self, exp_dict):
        super().__init__()
        self.exp_dict = exp_dict
        self.model = torch.nn.Sequential(torch.nn.Linear(46, 128, bias=False), torch.nn.BatchNorm1d(128), torch.nn.ReLU(),
                                                         torch.nn.Linear(128, 128, bias=False),
                                                         torch.nn.BatchNorm1d(128),
                                                         torch.nn.ReLU(), torch.nn.Linear(128, 2))
        if self.exp_dict["weights"] is not None:
            self.load_state_dict(torch.load(self.exp_dict["weights"]))

        self.generator = get_backbone(exp_dict["generator"]).eval()
        weights = torch.load(exp_dict["generator"]["weights"])
        self.generator.char_embedding.load_state_dict(weights["char_embedding"])
        self.generator.font_embedding.load_state_dict(weights["font_embedding"])


    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])


class ResNet(torch.nn.Module):

    def __init__(self, exp_dict):
        super().__init__()

        self.exp_dict = exp_dict
        self.model = get_backbone(exp_dict)

        if self.exp_dict["weights"] is not None:
            self.load_state_dict(torch.load(self.exp_dict["weights"]))


    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
