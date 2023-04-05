import torch
from .backbones import get_backbone

class Encoder(torch.nn.Module):

    def __init__(self, exp_dict):
        """ Constructor
        Args:
            model: architecture to train
            self.exp_dict: reference to dictionary with the global state of the application
        """
        super().__init__()
        self.model = get_backbone(exp_dict)
        self.exp_dict = exp_dict

    def load_state_dict(self, state_dict):

        self.model.load_state_dict(state_dict["model"])
