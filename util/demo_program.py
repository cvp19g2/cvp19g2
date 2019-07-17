import cv2
import numpy
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from cycle_gan.models import create_model
from cycle_gan.options.test_options import TestOptions
from cycle_gan.util.util import tensor2im
from util.ImageResizer import resizeAndPad
from util.resize_images import detect_single_face_dlib, image_resize, make_square

cam = cv2.VideoCapture(0)

cv2.namedWindow("GAN Demo")

haar_cascade_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img_counter = 0

opt = TestOptions().parse()  # get test option
# hard-code some parameters for test
opt.num_threads = 1  # test code only supports num_threads = 1
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1
opt.model = "cycle_gan"
opt.name = "maps_cyclegan"
opt.no_dropout = "true"
# no visdom display; the test code saves the results to a HTML file
model = create_model(opt)  # create a model given opt.model and other options
model.setup(opt)

def f(x):
    return np.add(128,np.multiply(x, 128))

def getImages(image):
    input = {}

    img_to_pad = transforms.Compose([
        transforms.Pad(padding=2, padding_mode="constant", fill=0),
        transforms.ToTensor(),
    ])

    input["A"] = img_to_pad(image).unsqueeze(0)
    input["B"] = img_to_pad(image).unsqueeze(0)
    input["A_paths"] = ""
    input["B_paths"] = ""
    model.set_input(input)  # unpack data from data loader
    model.test()         # run inference
    return model.get_current_visuals()  # get image results

while True:
    ret, frame = cam.read()
    cv2.imshow("GAN Demo", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img = frame
        face = detect_single_face_dlib(img)

        if face != None:

            print("Face found")

            new_img = img[face[1]:(face[1] + face[3]), face[0]:(face[0] + face[2])]
            new_img = image_resize(new_img, height=128)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            resized_img = make_square(Image.fromarray(new_img))

            #cv2.imwrite("sam_%d.jpg"%img_counter, resized_img)
            
            results = getImages(resized_img)

            image_numpy_fakeB = resizeAndPad(tensor2im(results["fake_B"]), (128, 128), 0)
            image_numpy_recA = resizeAndPad(tensor2im(results["rec_B"]), (128, 128), 0)

            image_numpy_fakeB = cv2.cvtColor(image_numpy_fakeB, cv2.COLOR_RGB2BGR)
            image_numpy_recA = cv2.cvtColor(image_numpy_recA, cv2.COLOR_RGB2BGR)
            resized_img  = cv2.cvtColor(numpy.array(resized_img), cv2.COLOR_RGB2BGR)

            numpy_horizontal = np.hstack((resized_img, image_numpy_fakeB, image_numpy_recA))
            
            cv2.imshow("GAN Demo", numpy_horizontal)

        escaped = False
        while not escaped:
            k = cv2.waitKey(1)
            if k%256 == 27:
                escaped = True


cam.release()

cv2.destroyAllWindows()