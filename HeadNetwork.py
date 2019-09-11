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

x_train = []
y_train = []

for filename in os.listdir("Test/head_nparray"):
    file = np.load("Test/head_nparray/" + filename)
    x = file["x"]
    y = file["y"]
    x_train.append(x)
    y_train.append(y)

    eye = (y[0], y[1])
    finn = (y[2], y[3])
    minX = y[4]
    minY = y[5]
    maxX = y[6]
    maxY = y[7]

    #cv2.rectangle(x, (minX, minY), (maxX, maxY), (0, 255, 0), 5)
    #cv2.circle(x, eye, 2, (255, 0, 0), 5)
    #cv2.circle(x, finn, 2, (0, 0, 255), 5)
    #cv2.imshow(filename, x)
    #cv2.waitKey(0)

x_train = np.array(x_train)
y_train = np.array(y_train)

model = keras.models.Sequential()
print(x_train.shape)

basefilter = 8

model.add(Conv2D(basefilter, 3, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(250, 684, 3)))
model.add(MaxPooling2D())
model.add(keras.layers.BatchNormalization())

model.add(Conv2D(basefilter * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPooling2D())
model.add(keras.layers.BatchNormalization())

model.add(Conv2D(basefilter * 3, 3, activation='relu', padding='same', kernel_initializer='he_normal'))
model.add(MaxPooling2D())
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(256, activation="relu"))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(8, activation="relu"))

adam = keras.optimizers.adam(lr=8e-5)
model.compile(optimizer=adam, loss="mse", metrics=["accuracy"])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto', restore_best_weights=True)

def lr_scheduler(epoch, lr):
    decay_rate = 0.99
    decay_step = 20
    if (epoch + 1) % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

decay = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)

history = model.fit(x_train, y_train, epochs=100, shuffle=True, validation_split=0.2, callbacks=[early_stop, decay])

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['vgg_train', 'vgg_val', 'my_train', 'my_val'], loc='upper left')
plt.show()

testImage = np.load("Test/head_val/25427 070420.9-cam2-000111.png.npz")["x"]
pred = model.predict(testImage.reshape(1, 250, 684, 3))

resultData = pred[0]



eye = (resultData[0], resultData[1])
finn = (resultData[2], resultData[3])
headXYMin = (resultData[4], resultData[5])
headXYMax = (resultData[6], resultData[7])
print(eye)
cv2.circle(testImage, eye, 2, (0, 0, 255), 10)
cv2.circle(testImage, finn, 2, (0, 255, 0), 10)
cv2.rectangle(testImage, headXYMin, headXYMax, (0, 255, 0), 1)
cv2.imshow("result", testImage)
cv2.waitKey(0)