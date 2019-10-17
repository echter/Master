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
import random
import os

image_size = 128

x_train = []
y_train = []

print("Loading data...")
for filename in os.listdir("Test/head_val"):
    file = np.load("Test/head_val/" + filename)
    x = file["x"]
    y = file["y"]
    x = cv2.resize(x, (image_size, image_size))
    #print(x.shape)
    x_train.append(x)
    y_train.append(np.array([y[0], y[1]]))#, y[2], y[3], y[4], y[5], y[6], y[7]]))
    #y_train.append(np.array([y[0], y[1]]))#, y[2], y[3], y[4], y[5], y[6], y[7]]))
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

def get_input(path):

    image = np.load(("{}").format(path))["x"]
    return image

def get_output(path):

    image = np.load(("{}").format(path))["y"]
    return image

def custom_generator(files, batch_size=1):

    while True:

        batch_path = []#np.random.choice(files, batch_size)

        for i in range(batch_size):
            batch_path.append(random.choice(files))

        batch_input = []
        batch_output = []

        for input_path in batch_path:

            x = get_input(input_path)
            y = get_output(input_path)
            x = cv2.resize(x, (image_size, image_size))
            x = np.array(x).astype(np.uint8)

            batch_input.append(x)
            batch_output.append(np.array([y[0], y[1]]))


        batch_x = np.array(batch_input).reshape(-1, image_size, image_size, 3)
        batch_y = np.array(batch_output)

        yield (batch_x, batch_y)

x_train = np.array(x_train)
y_train = np.array(y_train)

print("Starting model...")

model = keras.models.Sequential()
print(x_train.shape)

shape = (image_size, image_size, 3)

basefilter = 16

base_model = keras.applications.inception_v3.InceptionV3(include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='relu')(x)

model = Model(inputs=base_model.input, outputs=predictions)

names = []
for filename in os.listdir("Test/overfit_test"):
    names.append(("Test/overfit_test/{}").format(filename))

#model.compile(optimizer="adam", loss=keras.losses.mean_absolute_error, metrics=["accuracy"])
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto', restore_best_weights=True)

checkpointer = ModelCheckpoint("eye_finder_9000.sav", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

def lr_scheduler(epoch, lr):
    decay_rate = 0.95
    decay_step = 10
    if (epoch + 1) % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

decay = keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)

#history = model.fit(x_train, y_train, batch_size=1, epochs=100, shuffle=True, callbacks=[decay])
history = model.fit_generator(custom_generator(names, batch_size=16), validation_data=(x_train, y_train), steps_per_epoch=1, epochs=100, callbacks=[])


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['vgg_train', 'vgg_val', 'my_train', 'my_val'], loc='upper left')
plt.show()

testImage = np.load("Test/head_val/25427 070420.9-cam2-000111.png.npz")["x"]
testImage = cv2.resize(testImage, (image_size, image_size))
pred = model.predict(testImage.reshape(1, image_size, image_size, 3))

resultData = pred[0]



eye = (int(resultData[0]), int(resultData[1]))
#finn = (resultData[2], resultData[3])
#headXYMin = (resultData[4], resultData[5])
#headXYMax = (resultData[6], resultData[7])
print(eye)
print(np.load("Test/head_val/25427 070420.9-cam2-000111.png.npz")["y"])
cv2.circle(testImage, eye, 2, (0, 0, 255), 2)
#cv2.circle(testImage, finn, 2, (0, 255, 0), 10)
#cv2.rectangle(testImage, headXYMin, headXYMax, (0, 255, 0), 1)
cv2.imshow("result", testImage)
cv2.waitKey(0)