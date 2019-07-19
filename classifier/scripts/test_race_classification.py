from keras import models, backend
import cv2
import numpy as np
import os
import tensorflow as tf
import keras

keras.backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))
middle = models.load_model("../weights/vgg16_5race_15epoch.hdf5", custom_objects={"backend": backend})

file_path = "../data/img_align_celeba"
# file_path = "../data/data_race/val/indian"
files = sorted(os.listdir(file_path))
predicted_path = '../data/data_race_predicted'


def get_folder_name(race):
    return {
        0: 'asian',
        1: 'black',
        2: 'indian',
        3: 'others',
        4: 'white'
    }[race]


print(len(files))

for file in files:
    image1 = cv2.imread(os.path.join(file_path, file))
    image1 = cv2.resize(image1, (224, 224))
    embedding1 = middle.predict(np.expand_dims(image1 / 255., axis=0))[0]
    embedding2 = middle.predict(np.expand_dims(np.fliplr(image1) / 255., axis=0))[0]

    pred = (embedding1 + embedding2) / 2
    max_score = max(pred)
    race = np.where(pred == max_score)[0][0]
    
    if max_score > 0.96 and race in [0]:
        print('filename: {}, prediction:{}, score: {}'.format(file, get_folder_name(race), max_score))
        # cv2.imwrite(os.path.join(predicted_path, '{}_{}'.format(str(race), file)), image1)
        # os.remove(os.path.join(file_path, file))
        cv2.imshow('image', image1)
        cv2.waitKey()
