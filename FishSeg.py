import json

import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input, Dropout, concatenate
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
import os

def generalized_dice_loss_w(y_true, y_pred):

    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = K.sum(numerator,(0,1,2))
    numerator = K.sum(numerator)

    denominator = y_true+y_pred
    denominator = K.sum(denominator,(0,1,2))
    denominator = K.sum(denominator)

    gen_dice_coef = numerator/denominator

    return 1-2*gen_dice_coef

x_train = []
y_train = []

size = 512
counter = 0

for filename in os.listdir("Test/cropped"):
    x = np.load("Test/cropped/" + filename)["x"]
    y = np.load("Test/cropped/" + filename)["y"]

    x = x.reshape(size, size, 3)
    y = y.reshape(size, size, 1)

    x_train.append(x)
    y_train.append(y)

    counter += 1

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = x_train.reshape(counter, size, size, 3)
y_train = y_train.reshape(counter, size, size, 1)

inputs = Input((size, size, 3))

model = keras.models.Sequential()

baseFilter = 64
dropout = 0.50

conv1 = Conv2D(baseFilter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
conv1 = Conv2D(baseFilter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(baseFilter*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
conv2 = Conv2D(baseFilter*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(baseFilter*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
conv3 = Conv2D(baseFilter*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(baseFilter*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
conv4 = Conv2D(baseFilter*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
drop4 = Dropout(dropout)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(baseFilter*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
conv5 = Conv2D(baseFilter*16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
drop5 = Dropout(dropout)(conv5)

up6 = Conv2D(baseFilter*8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
merge6 = concatenate([drop4, up6], axis=3)
conv6 = Conv2D(baseFilter*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
conv6 = Conv2D(baseFilter*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

up7 = Conv2D(baseFilter*4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
merge7 = concatenate([conv3, up7], axis=3)
conv7 = Conv2D(baseFilter*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
conv7 = Conv2D(baseFilter*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

up8 = Conv2D(baseFilter*2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
merge8 = concatenate([conv2, up8], axis=3)
conv8 = Conv2D(baseFilter*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
conv8 = Conv2D(baseFilter*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

up9 = Conv2D(baseFilter, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
merge9 = concatenate([conv1, up9], axis=3)
conv9 = Conv2D(baseFilter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
conv9 = Conv2D(baseFilter, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

model = Model(input=inputs, output=conv10)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto', restore_best_weights=True)

checkpointer = keras.callbacks.ModelCheckpoint("spot_segmenter_9000.sav", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

def lr_scheduler(epoch, lr):
    decay_rate = 0.99
    decay_step = 10
    if (epoch + 1) % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

decay = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)

model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=Adam(lr=1e-5), loss=generalized_dice_loss_w, metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=4, epochs=100, validation_split=0.2, callbacks=[decay, early_stop, checkpointer])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['vgg_train', 'vgg_val', 'my_train', 'my_val'], loc='upper left')
plt.show()

image = np.load("Test/val/cropped_v5.png.npz")["x"]
image = image.reshape(1, size, size, 3)
pred = model.predict(image)
pred = pred.reshape((size, size))
plt.imshow(pred)
plt.show()