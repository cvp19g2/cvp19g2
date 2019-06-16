import torch
from math import sqrt
from torch import nn
from torchvision import models

vgg16 = models.vgg16_bn(pretrained=True)
pairwiseDistance = nn.PairwiseDistance(p=2)

print(vgg16.features)

result1 = vgg16.features(torch.randn(1, 3, 224, 224))
result2 = vgg16.features(torch.randn(1, 3, 224, 224))

print(result1.size())

print(result1[0][500][0][0]);
distance = pairwiseDistance(result1, result2)

result = 0
for i in range(len(result1[0])):
    for j in range(len(result1[0][0])):
        for h in range(len(result1[0][0][0])):
            result += pow(result1[0][i][j][h] - result2[0][i][j][h], 2)
result = sqrt(result)

print(result)