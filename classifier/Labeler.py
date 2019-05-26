from __future__ import print_function, division

from shutil import copyfile

import numpy
import torch
import torch.nn as nn
from PIL import Image
import os
from torchvision import models, transforms

vgg16 = models.vgg16_bn()
vgg16.load_state_dict(torch.load("../input/vgg16bn/vgg16_bn.pth"))

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 4)]) # Add our layer with 4 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

vgg16.load_state_dict(torch.load('../classifier/VGG16_v2-UTK.pt'))

imsize = 224
loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

vgg16.cuda()

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()

imageNumber = 0

for filename in os.listdir("../data/celebA/img_align_celeba/"):
    image =  image_loader("../data/celebA/img_align_celeba/%s" % filename)
    output = vgg16(image)
    maxClass = numpy.argmax(output)
    maxValue = numpy.max(output)

    if maxValue > 0.6:
        copyfile("../data/celebA/img_align_celeba/%s" % filename, "../data/celebA/high_confidence/%s" % ("face_%d_%d" % (imageNumber, maxClass)))

    imageNumber = imageNumber + 1

    if imageNumber % 1000 == 0:
        print(imageNumber)