import cv2
import numpy as np
from ImageResizer import resizeAndPad

cam = cv2.VideoCapture(0)

cv2.namedWindow("GAN Demo")

haar_cascade_face = cv2.CascadeClassifier("util/haarcascade_frontalface_default.xml")

img_counter = 0

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
            resized_img = resizeAndPad(new_img, (400, 400), 0)
            
            numpy_horizontal = np.hstack((resized_img, resized_img, resized_img))
            
            cv2.imshow("GAN Demo", numpy_horizontal)
            break

        escaped = False
        while not escaped:
            k = cv2.waitKey(1)
            if k%256 == 27:
                escaped = True


cam.release()

cv2.destroyAllWindows()