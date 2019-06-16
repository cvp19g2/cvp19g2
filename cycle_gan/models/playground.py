from torch import nn
from torchvision import models

vgg16 = models.vgg16_bn(pretrained=True)
features = list(vgg16.children())[:-2]
vgg16.features = nn.Sequential(*features)

print(vgg16.features)