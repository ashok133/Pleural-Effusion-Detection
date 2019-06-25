"""
@author - Ashok Patel
"""

import h5py
import math
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import scipy
import base64
import io
from PIL import Image
from scipy import ndimage
from utils import *
from tensorflow.python.framework import ops
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform

def img_to_base64(image_loc):
    with open(image_loc, "rb") as image_file:
        image_read = image_file.read()
        encoded_image = base64.encodestring(image_read)
    save_and_predict(encoded_image)

def save_and_predict(img64):
    temp_rand = np.random.randn()
    file_loc = "tested_images/" + str(temp_rand) + ".png"
    # imgdata = base64.b64decode(img64)
    with open(file_loc, "wb") as fp:
        fp.write(base64.b64decode(img64))
    prediction = predict_status(file_loc)
    return prediction

def img_to_numpy_array(img_object_loc):
    img = load_img(img_object_loc, grayscale=True)
    img.thumbnail((64, 64))
    x = img_to_array(img)
    x = x.reshape((1, 64, 64, 1))
    return x

def predict_status(image_loc):
    predicted_status = 0
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model = load_model('learned_model/model.h5')
        img_np_array = img_to_numpy_array(image_loc)
        if model.predict(img_np_array)[0][0] > 0.5:
            predicted_status = 1
        print("The network predicts a sign of :  " + str(predicted_status))
        return str(predicted_status)

predict_status('tested_images/test.png')
