import torch
from torch import nn
from torchvision import models

vgg16 = models.vgg16_bn(pretrained=True)

print(vgg16.features)

result = vgg16.features(torch.randn(1, 3, 224, 224))

print(result.size())