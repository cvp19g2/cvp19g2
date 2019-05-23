import os
from shutil import copyfile

numProcessedImages = 0

for filename in os.listdir("../data/UTKFace/"):
    labels = filename.split("_")
    ethnicity = labels[2]
    folder = "others"
    if ethnicity == "0":
        folder = "white"
    elif ethnicity == "1":
        folder = "black"
    elif ethnicity == "2":
        folder = "asian"
    elif ethnicity == "3":
        folder = "indian"

    copyfile("../data/UTKFace/%s" % filename, "../data/utk_split/%s/%s" % (folder, filename))
    numProcessedImages = numProcessedImages + 1

    if (numProcessedImages % 100 == 0):
        print(numProcessedImages)



numProcessedImages = 0
for filename in os.listdir("../data/UTKFace/"):
    if (numProcessedImages < 20000):
        copyfile("../data/UTKFace/%s" % filename, "../data/utk_training/TRAIN/%s" % filename)
    elif (numProcessedImages < 20500):
        copyfile("../data/UTKFace/%s" % filename, "../data/utk_training/TEST/%s" % filename)
    else:
        copyfile("../data/UTKFace/%s" % filename, "../data/utk_training/VAL/%s" % filename)
    numProcessedImages = numProcessedImages + 1



