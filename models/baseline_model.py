
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class EfficientNetBinary(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT,
        )

        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)
