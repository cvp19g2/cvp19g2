import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

from cycle_gan.models import create_model
from cycle_gan.options.test_options import TestOptions
from cycle_gan.util import util
from cycle_gan.util.util import tensor2im
from util.ImageResizer import resizeAndPad

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

    input["A"] = img_to_pad(Image.fromarray(image)).unsqueeze(0)
    input["B"] = img_to_pad(Image.fromarray(image)).unsqueeze(0)
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_rects = haar_cascade_face.detectMultiScale(gray, scaleFactor=1.1)

        for (x,y,w,h) in faces_rects:

            enlarge = 1.5

            width = int(w * enlarge)
            height = int(h * enlarge)

            newX = int(max(0, x - 0.25*w))
            newY = int(max(0, y - 0.25*h))

            print("Face found")
            new_img = img[newY:(newY+height), newX:(newX+width)]
            resized_img = resizeAndPad(new_img, (128, 128), 0)

            cv2.imwrite("sam_%d.jpg"%img_counter, resized_img)
            
            results = getImages(resized_img)

            image_numpy_fakeB = resizeAndPad(tensor2im(results["fake_B"]), (128, 128), 0)
            image_numpy_recA = resizeAndPad(tensor2im(results["rec_A"]), (128, 128), 0)

            numpy_horizontal = np.hstack((resized_img, image_numpy_fakeB, image_numpy_recA))
            
            cv2.imshow("GAN Demo", numpy_horizontal)
            break

        escaped = False
        while not escaped:
            k = cv2.waitKey(1)
            if k%256 == 27:
                escaped = True


cam.release()

cv2.destroyAllWindows()