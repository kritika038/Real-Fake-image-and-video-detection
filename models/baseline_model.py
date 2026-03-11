# import torch
# import torch.nn as nn
# from torchvision import models


# class EfficientNetBinary(nn.Module):

#     def __init__(self):
#         super(EfficientNetBinary, self).__init__()

#         self.model = models.efficientnet_b0(pretrained=True)

#         in_features = self.model.classifier[1].in_features

#         # 2 outputs: [REAL, FAKE]
#         self.model.classifier[1] = nn.Linear(in_features, 2)

#     def forward(self, x):
#         return self.model(x)
import torch
import torch.nn as nn
from torchvision import models


class EfficientNetBinary(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.efficientnet_b0(pretrained=True)

        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(in_features, 2)


    def forward(self, x):

        return self.model(x)