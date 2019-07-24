import cv2
import numpy
import numpy as np
from PIL import Image
from torchvision.transforms import transforms

from cycle_gan.models import create_model
from cycle_gan.options.test_options import TestOptions
from cycle_gan.util.util import tensor2im
from util.ImageResizer import resizeAndPad
from util.resize_images import detect_single_face_dlib, image_resize, make_square
import os
from scipy.misc import imresize
from cycle_gan.data.base_dataset import get_transform
from cycle_gan.data import create_dataset
import sys

square_length = 300

opt = TestOptions().parse()  # get test option
# hard-code some parameters for test
opt.num_threads = 0  # test code only supports num_threads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1
#opt.dataroot = ".\tmp\"
#opt.checkpoints_dir = ".\cycle_gan\checkpoints\"
opt.model = "cycle_gan"
opt.dataset_mode = "unaligned"
opt.phase = "test"
opt.no_dropout = "true"
# no visdom display; the test code saves the results to a HTML file
model = create_model(opt)  # create a model given opt.model and other options
model.setup(opt)

pathIn = "E:\TMP\CVP19G2\Demo\Input"
pathOut = "E:\TMP\CVP19G2\Demo\Output"

for file in os.listdir(pathIn):
    #print("Processing: " + file)

    img = cv2.imread(os.path.join(pathIn, file))
    face = detect_single_face_dlib(img)

    if face != None:

        print("Face found")

        new_img = img[face[1]:(face[1] + face[3]), face[0]:(face[0] + face[2])]
        new_img = image_resize(new_img, height=128)
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        resized_img = make_square(Image.fromarray(new_img))

        resized_img.save(os.path.join("./tmp/testA", file))
        resized_img.save(os.path.join("./tmp/testB", file))

        dataset = create_dataset(opt)

        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results

            image_numpy_fakeB = imresize(tensor2im(visuals["fake_B"]), (square_length, square_length), interp='bicubic')
            image_numpy_recA = imresize(tensor2im(visuals["rec_A"]), (square_length, square_length), interp='bicubic')

            image_numpy_fakeB = cv2.cvtColor(image_numpy_fakeB, cv2.COLOR_RGB2BGR)
            image_numpy_recA = cv2.cvtColor(image_numpy_recA, cv2.COLOR_RGB2BGR)

            resized_img = imresize(resized_img, (square_length, square_length), interp='bicubic')
            resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)

            numpy_horizontal = np.hstack((resized_img, image_numpy_fakeB, image_numpy_recA))

            cv2.imwrite(os.path.join(pathOut, file), numpy_horizontal)
        
        os.remove(os.path.join("./tmp/testA", file))
        os.remove(os.path.join("./tmp/testB", file))

    else:
        print("No face found for: " + file)

    #print("Done: " + file)
    