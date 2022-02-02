import torch
import torchvision


class ResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feat_extract = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.feat_extract(x)
