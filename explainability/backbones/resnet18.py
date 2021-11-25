import torch
import torchvision


class ResNet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feat_extract = torchvision.models.resnet18(pretrained=False)
        self.feat_extract.fc = torch.nn.Identity()
        self.output_size = 512

    def forward(self, x):
        return self.feat_extract(x)
