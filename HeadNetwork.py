import json

import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input, Dropout, concatenate, Dense, ZeroPadding2D, Convolution2D, Flatten, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
import os

x_train = []
y_train = []

print("Loading data...")
for filename in os.listdir("Test/head_nparray"):
    file = np.load("Test/head_nparray/" + filename)
    x = file["x"]
    y = file["y"]
    x_train.append(x)
    y_train.append(np.array([y[0], y[1]]))#, y[2], y[3], y[4], y[5], y[6], y[7]]))
    #print(y)
    #print(np.array([y[0], y[1], y[2], y[3]]))

    eye = (int(y[0] * 684), int(y[1] * 250))
    finn = (y[2], y[3])
    minX = y[4]
    minY = y[5]
    maxX = y[6]
    maxY = y[7]

    #print(eye)

    #cv2.rectangle(x, (minX, minY), (maxX, maxY), (0, 255, 0), 5)
    #cv2.circle(x, eye, 2, (255, 0, 0), 5)
    #cv2.circle(x, finn, 2, (0, 0, 255), 5)
    #cv2.imshow("y", x)
    #print(filename)
    #cv2.waitKey(0)

x_train = np.array(x_train)
y_train = np.array(y_train)

print(y_train)

print("Starting model...")

model = keras.models.Sequential()
print(x_train.shape)

shape = (250, 684, 3)

basefilter = 16

model = keras.models.Sequential()
model.add(Convolution2D(basefilter, 3, activation='relu', padding="same", input_shape=shape))
model.add(Convolution2D(basefilter, 3, activation='relu', padding="same"))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='relu'))

model.compile(optimizer=Adam(lr=1e-6), loss="mse", metrics=["accuracy"])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto', restore_best_weights=True)

checkpointer = ModelCheckpoint("eye_finder_9000.sav", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

def lr_scheduler(epoch, lr):
    decay_rate = 0.99
    decay_step = 10
    if (epoch + 1) % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

decay = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)

history = model.fit(x_train, y_train, batch_size=1, epochs=100, shuffle=True, callbacks=[decay])

# summarize history for loss
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['vgg_train', 'vgg_val', 'my_train', 'my_val'], loc='upper left')
plt.show()

testImage = np.load("Test/head_val/25427 070420.9-cam2-000111.png.npz")["x"]
pred = model.predict(testImage.reshape(1, 250, 684, 3))

resultData = pred[0]



eye = (int(resultData[0]), int(resultData[1]))
#finn = (resultData[2], resultData[3])
#headXYMin = (resultData[4], resultData[5])
#headXYMax = (resultData[6], resultData[7])
print(eye)
cv2.circle(testImage, eye, 2, (0, 0, 255), 2)
#cv2.circle(testImage, finn, 2, (0, 255, 0), 10)
#cv2.rectangle(testImage, headXYMin, headXYMax, (0, 255, 0), 1)
cv2.imshow("result", testImage)
cv2.waitKey(0)