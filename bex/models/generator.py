import torch
from .backbones import get_backbone


class Generator(torch.nn.Module):

    def __init__(self, exp_dict):
        """ Constructor
        Args:
            model: architecture to train
            self.exp_dict: reference to dictionary with the global state of the application
        """
        super().__init__()
        if "generator_dict" in exp_dict:
            self.exp_dict = exp_dict["generator_dict"]
        else:
            self.exp_dict = exp_dict
        self.model = get_backbone(self.exp_dict)

        self.load_state_dict(torch.load(self.exp_dict["weights"]))

    def load_state_dict(self, state_dict):

        self.model.load_state_dict(state_dict["generator"])
        self.model.char_embedding.load_state_dict(state_dict["char_embedding"])
        self.model.font_embedding.load_state_dict(state_dict["font_embedding"])
