import torch
from torch import nn
from torchvision import models

vgg16 = models.vgg16_bn(pretrained=True)
vgg16.cuda()
pairwiseDistance = nn.PairwiseDistance(p=2)

print(vgg16.features)

result1 = vgg16.features(torch.randn(1, 3, 224, 224).cuda())
result2 = vgg16.features(torch.randn(1, 3, 224, 224).cuda())

distance = torch.dist(result1, result2, 2)

print(distance)