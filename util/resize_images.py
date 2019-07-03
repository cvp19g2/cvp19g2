import dlib
import matplotlib.pyplot as plt
import cv2
import os
import sys
import getopt
from PIL import Image

#https://hackernoon.com/gender-and-race-change-on-your-selfie-with-neural-nets-9a9a1c9c5c16
def detect_single_face_dlib(img_rgb, rescale=(1.1, 1.5, 1.1, 1.3)):
    fd_front_dlib = dlib.get_frontal_face_detector()
    face = fd_front_dlib(img_rgb, 1)
    if len(face) > 0:
        face = sorted([(t.width() * t.height(), (t.left(), t.top(), t.width(), t.height()))
                       for t in face],
                      key=lambda t: t[0], reverse=True)[0][1]
    else:
        return None

    if rescale is not None and face is not None:
        if type(rescale) != tuple:
            rescale = (rescale, rescale, rescale, rescale)
        (x, y, w, h) = face

        w = min(img_rgb.shape[1] - x, int(w / 2 + rescale[2] * w / 2))
        h = min(img_rgb.shape[0] - y, int(h / 2 + rescale[3] * h / 2))

        fx = max(0, int(x + w / 2 * (1 - rescale[0])))
        fy = max(0, int(y + h / 2 * (1 - rescale[1])))
        fw = min(img_rgb.shape[1] - fx, int(w - w / 2 * (1 - rescale[0])))
        fh = min(img_rgb.shape[0] - fy, int(h - h / 2 * (1 - rescale[1])))

        face = (fx, fy, fw, fh)
    return face

#Resizing image and filling background black
def make_square(im, min_size=128, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    a, b = (int((size - x) / 2), int((size - y) / 2))
    new_im.paste(im, (a,b))
    return new_im

#https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def main(argv):
    pathIn = ""
    pathOut = ""
    
    opts, args = getopt.getopt(argv, "i:o:",["inpath=","outpath="])
    
    for opt, arg in opts:
        if opt in ("-i", "--inpath"):
            pathIn = arg
        elif opt in ("-o", "--outpath"):
            pathOut = arg
    
    if pathIn == "" or pathOut == "":
        print("Path Error")
        sys.exit()
    
    for file in os.listdir(pathIn):
        #print("Processing: " + file)
    
        img = cv2.imread(os.path.join(pathIn, file))
        face = detect_single_face_dlib(img)
        
        if face != None:
            new_img = img[face[1]:(face[1]+face[3]), face[0]:(face[0]+face[2])]
            new_img = image_resize(new_img, height = 128)
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
            resized_img = make_square(Image.fromarray(new_img))

            x, y = resized_img.size

            if x is 128 and y is 128:
                resized_img.save(os.path.join(pathOut, file))
            else:
                print("Wrong size for: " + file)
        else:
            print("No face found for: " + file)

        #print("Done: " + file)

if __name__== "__main__":
    main(sys.argv[1:])
