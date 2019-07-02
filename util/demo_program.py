import cv2
import numpy as np

cam = cv2.VideoCapture(0)

cv2.namedWindow("GAN Demo")

haar_cascade_face = cv2.CascadeClassifier("util/haarcascade_frontalface_default.xml")

img_counter = 0
width = 221
height = 221

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
        print('Faces found: ', len(faces_rects))

        for (x,y,w,h) in faces_rects:
            print("Face found")
            new_img = img[y:(y+h), x:(x+w)]
            resized_img = cv2.resize(new_img, (width, height))
            
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