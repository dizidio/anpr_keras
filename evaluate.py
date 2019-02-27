import numpy as np
import gen
import itertools
import cv2

import keras
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, LSTM, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import time
import os
import cv2
import random

np.random.seed(1120)
random.seed(1120)

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS


## LIMIT GPU MEMORY
config = tf.ConfigProto()
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

def diff_letters(a,b):
    return sum ( a[i] != b[i] for i in range(len(a)) )

def unzip(b):
    xs, ys = zip(*b)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

def code_to_vec(p, code):
    def char_to_vec(c):
        y = np.zeros((len(CHARS),))
        y[CHARS.index(c)] = 1.0
        return y

    c = np.vstack([char_to_vec(c) for c in code])

    return np.concatenate([[1. if p else 0], c.flatten()])

def letter_probs_to_code(letter_probs):
    return "".join(CHARS[i] for i in np.argmax(letter_probs, axis=1))      



###########################################################

path = "./dataset_210618_test/"
imgs_png = [f for f in os.listdir(path) if f.endswith('.png')]

model = load_model('Model20190227_120755.h5')

erros_digitos = 0;
total_digitos = 0;

for img in imgs_png:
    im = cv2.imread(path+img)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = im_gray.reshape([1,64,128,1])
    im_gray = keras.applications.nasnet.preprocess_input(im_gray)
    results = model.predict(im_gray).ravel()
    results = results[1:].reshape([7,len(CHARS)]);
    code = letter_probs_to_code(results)
    print("Real {} - Pred {}".format(img[-13:-6], code))
    erros_digitos += diff_letters(code, img[-13:-6]);
    total_digitos += 7;
    

acc_digitos = (total_digitos - erros_digitos)/(total_digitos);
print(acc_digitos)

