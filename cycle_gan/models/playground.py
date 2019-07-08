import torch
from torch import nn
from torchvision import models


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


vgg16 = models.vgg16_bn(pretrained=True)
vgg16.cuda()
pairwiseDistance = nn.PairwiseDistance(p=2)

result1 = vgg16.features(torch.randn(1, 3, 224, 224).cuda())

print(gram_matrix(result1))
