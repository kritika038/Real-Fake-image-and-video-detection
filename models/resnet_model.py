import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNetBinary(nn.Module):

    def __init__(self):
        super(ResNetBinary, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        in_features = self.model.fc.in_features

        # 2 outputs: [REAL, FAKE]
        self.model.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)
