import numpy as np
import gen
import itertools
import cv2

import keras
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import time
import random

np.random.seed(1120)
random.seed(1120)

## LIMIT GPU MEMORY
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS

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

def read_batches(batch_size):
    g = gen.generate_ims()
    pre_processing = keras.applications.nasnet.preprocess_input
    def gen_vecs():
        for im, c, p in itertools.islice(g, batch_size):
            yield pre_processing(255.*im.reshape([64,128,1])), code_to_vec(p, c)
    while True:
        yield unzip(gen_vecs())
        
class print_codes(keras.callbacks.Callback):      
    def on_epoch_end(self, batch, logs={}):
        cont = 0;
        erros = 0;
        score_results = self.model.predict(val_batch[0])
        real_scores = val_batch[1];
        for i,j in zip(score_results,real_scores):
          probs1 = i[1:].reshape([7,36]);
          probs2 = j[1:].reshape([7,36]);
          code1 = letter_probs_to_code(probs1);
          code2 = letter_probs_to_code(probs2);
          erros = erros + diff_letters(code1,code2);
          cont = cont + 7;        
          #print("{:.2f} - {} <-> {} - {}".format(i[0], code1, j[0], code2));
        print("Acc: {}%".format(100*(cont-erros)/cont));
        

for x in read_batches(100):
  val_batch = x;
  break;

c_test = print_codes()

def lr_schedule(epoch):
  if (epoch<=10):
    return 1.0*(0.1**int(epoch));
  else:
    return 1.0*(0.1**int(epoch/10))


###########################################################


batch_size = 50
input_shape = (64, 128, 1)
learning_rate = 0.0005;
steps_per_epoch = 500;
epochs = 1000;

#model = Sequential()
#model.add(Conv2D(48, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape, data_format='channels_first', padding='same'))
#model.add(BatchNormalization())
##model.add(Conv2D(48, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
##model.add(Dropout(0.25))#

#model.add(Flatten())
#model.add(Dense(2048, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(253, activation='sigmoid'))

model = keras.applications.nasnet.NASNetMobile(input_shape=(64,128,1), include_top=True, weights=None, classes=253)

#model = keras.applications.mobilenet_v2.MobileNetV2(include_top = True, weights = None, input_shape = (64,128,1), classes=253)

model.layers[-1].activation = keras.activations.sigmoid
model.summary()

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
              
#reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=0.0001, verbose=1)

model.fit_generator(read_batches(batch_size), steps_per_epoch=steps_per_epoch, callbacks = [c_test], epochs=epochs, verbose=1)

timestamp = time.strftime("Model%Y%m%d_%H%M%S.h5")
model.save(timestamp)

print("Model saved at {}".format(timestamp));
