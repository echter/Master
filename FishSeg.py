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
from keras.preprocessing.image import ImageDataGenerator
import random

imageSize = size = 512

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

def combine_generator(gen1, gen2):
    while True:
        image = gen1.next()
        image2 = gen2.next()
        yield (image[:,:,:,0].reshape([image.shape[0], imageSize, imageSize, 1]), image2[:,:,:,0].reshape([image2.shape[0], imageSize, imageSize, 1]))

def get_input(path):

    image = np.load(("{}").format(path))["x"]
    return image

def get_output(path):

    image = np.load(("{}").format(path))["y"]
    return image

def custom_generator():
    # TODO only keep data which has more than x pixels. Currently some images with like 0 pixels are allowed which is bad

    args = dict(rotation_range=0,
        width_shift_range=0.01,
        height_shift_range=0.01,
        # rescale= 1. / 255,
        shear_range=0.05,
        zoom_range=0.05,
        vertical_flip=True,
        fill_mode='nearest')

    datagen = ImageDataGenerator(**args)

    while True:

        batch_path = []#np.random.choice(files, batch_size)

        #for i in range(0):#batch_size):
        #    batch_path.append(random.choice(files))

        batch_input = []
        batch_output = []

        size = 0

        # TODO make this select a bunch of random slices isntead of the whole set from 1 image
        for i in range(4):

            l_img = cv2.imread("spotless.png", 1)
            y_mask = np.zeros((l_img.shape[0], l_img.shape[1]))

            for i in range(0, random.randint(4, 13)):

                s_img = cv2.imread("dot2.png", -1)

                s_img[:, :, 0:2] = s_img[:, :, 0:2] * (random.randint(1, 100) / 70)

                reductionMultiplierX = (random.randint(1, 80) / 40) + 1
                reductionMultiplierY = (random.randint(1, 80) / 40) + 1

                s_img = cv2.resize(s_img, (
                int(s_img.shape[1] / reductionMultiplierX), int(s_img.shape[0] / reductionMultiplierY)))

                rows = cols = s_img.shape[0]

                # M = np.float32([[1, 0, 100], [0, 1, 50]])
                # dst = cv2.warpAffine(s_img, M, (cols, rows))

                # TODO ---------------- FIX THE CUTOFF

                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.randint(0, 360), 0.4)
                s_img = cv2.warpAffine(s_img, M, (cols, rows))

                s_img = cv2.resize(s_img, (int(s_img.shape[1] * 2.5), int(s_img.shape[0] * 2.5)))

                # cv2.imshow("partial", s_img)
                # cv2.waitKey(0)

                y_offset = random.randint(150, 400)
                x_offset = random.randint(300, 500)

                y1, y2 = y_offset, y_offset + s_img.shape[0]
                x1, x2 = x_offset, x_offset + s_img.shape[1]

                mask = s_img[:, :, 3] > 200

                for c in range(0, 2):
                    s_img[:, :, c] = s_img[:, :, c] * mask

                alpha_s = mask.astype(int)
                alpha_l = 1 - alpha_s

                l_cut = l_img[y1:y2, x1:x2, :]
                y_mask_cut = y_mask[y1:y2, x1:x2]

                s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY)
                kernel = np.ones((5, 5), np.float32) / 25
                s_img = cv2.filter2D(s_img, -1, kernel)

                # cv2.imshow("why", s_img)
                # cv2.waitKey(0)
                # print(s_img.shape)

                s_img = s_img.reshape(s_img.shape[1], s_img.shape[0], 1)

                if l_cut.shape[0] == l_cut.shape[1] == s_img.shape[0] == s_img.shape[1] == y_mask_cut.shape[0] == \
                        y_mask_cut.shape[1]:

                    simg = alpha_s * s_img[:, :, 0]

                    y_mask[y1:y2, x1:x2] += mask

                    for c in range(0, 3):
                        roi = alpha_l * l_cut[:, :, c]
                        l_img[y1:y2, x1:x2, c] = (simg + roi)

            l_img = cv2.resize(l_img, (512, 512))
            y_mask = cv2.resize(y_mask, (512, 512))

            #cv2.imshow("result", l_img)
            #cv2.imshow("mask", y_mask)
            #cv2.imwrite(("Showcase/{}.jpg").format(i), l_img)
            #cv2.waitKey(0)

            l_img = np.array(l_img).reshape(1, 512, 512, 3)
            y_mask = np.array(y_mask).reshape(1, 512, 512, 1)

            batch_input += [l_img]
            batch_output += [y_mask]
            size += 1

        batch_x = np.array(batch_input).reshape(-1, imageSize, imageSize, 3)
        batch_y = np.array(batch_output).reshape(-1, imageSize, imageSize, 1)

        yield (batch_x, batch_y)

def custom_generator_real(files, batch_size=4):

    visualise = False

    while True:

        batch_path = []#np.random.choice(files, batch_size)

        for i in range(batch_size):
            batch_path.append(random.choice(files))

        batch_input = []
        batch_output = []

        for input_path in batch_path:
            # print(input_path)

            input = get_input(input_path)
            output = get_output(input_path)
            output = output.reshape(512, 512, 1)

            output = output.astype(np.uint8)
            input = input.astype(np.uint8)

            translation_matrix = np.float32([ [1,0,random.randint(-30, 30)], [0,1,random.randint(-30, 30)]])

            input_t = cv2.warpAffine(input, translation_matrix, (512, 512))
            output_t = cv2.warpAffine(output, translation_matrix, (512, 512))

            if visualise:
                plt.subplot(211)
                plt.imshow(input_t)
                plt.subplot(212)
                plt.imshow(output_t)
                plt.show()

            batch_input.append(input_t)
            batch_output.append(output_t)

        batch_x = np.array(batch_input).reshape(-1, imageSize, imageSize, 3)
        batch_y = np.array(batch_output).reshape(-1, imageSize, imageSize, 1)
        batch_y = batch_y > 0.1
        batch_y = batch_y.astype(np.uint8)

        yield (batch_x, batch_y)

names = []
for filename in os.listdir("Test/cropped"):
    names.append(("Test/cropped/{}").format(filename))


x_train = []
y_train = []

x_val = []
y_val = []

counter = 0

for filename in os.listdir("Test/cropped"):
    x = np.load("Test/cropped/" + filename)["x"]
    y = np.load("Test/cropped/" + filename)["y"]

    x = x.reshape(size, size, 3)
    y = y.reshape(size, size, 1)

    x_train.append(x)
    y_train.append(y)

    counter += 1

for filename in os.listdir("Test/val"):
    x = np.load("Test/val/" + filename)["x"]
    y = np.load("Test/val/" + filename)["y"]

    x = x.reshape(size, size, 3)
    y = y.reshape(size, size, 1)

    x_val.append(x)
    y_val.append(y)

x_val = np.array(x_val)
y_val = np.array(y_val)

x_val = x_val.reshape(-1, size, size, 3)
y_val = y_val.reshape(-1, size, size, 1)

x_train = np.array(x_train)
y_train = np.array(y_train)

(xs, ys) = custom_generator_real(names).__next__()
for i in range(0, 0):

    x = xs[i]
    y = ys[i]

    print(x.shape)
    print(y.shape)

    plt.subplot(211)
    plt.imshow(x.reshape(512, 512, 3))
    plt.subplot(212)
    plt.imshow(y.reshape(512, 512))
    plt.show()

x_train = x_train.reshape(counter, size, size, 3)
y_train = y_train.reshape(counter, size, size, 1)

inputs = Input((size, size, 3))

model = keras.models.Sequential()

baseFilter = 64
dropout = 0.2

conv1 = Conv2D(baseFilter, 3, activation='relu', padding='same', kernel_initializer='normal')(inputs)
conv1 = Conv2D(baseFilter, 3, activation='relu', padding='same', kernel_initializer='normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(baseFilter*2, 3, activation='relu', padding='same', kernel_initializer='normal')(pool1)
conv2 = Conv2D(baseFilter*2, 3, activation='relu', padding='same', kernel_initializer='normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(baseFilter*4, 3, activation='relu', padding='same', kernel_initializer='normal')(pool2)
conv3 = Conv2D(baseFilter*4, 3, activation='relu', padding='same', kernel_initializer='normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(baseFilter*8, 3, activation='relu', padding='same', kernel_initializer='normal')(pool3)
conv4 = Conv2D(baseFilter*8, 3, activation='relu', padding='same', kernel_initializer='normal')(conv4)
drop4 = Dropout(dropout)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(baseFilter*16, 3, activation='relu', padding='same', kernel_initializer='normal')(pool4)
conv5 = Conv2D(baseFilter*16, 3, activation='relu', padding='same', kernel_initializer='normal')(conv5)
drop5 = Dropout(dropout)(conv5)

up6 = Conv2D(baseFilter*8, 2, activation='relu', padding='same', kernel_initializer='normal')(UpSampling2D(size=(2, 2))(drop5))
merge6 = concatenate([drop4, up6], axis=3)
conv6 = Conv2D(baseFilter*8, 3, activation='relu', padding='same', kernel_initializer='normal')(merge6)
conv6 = Conv2D(baseFilter*8, 3, activation='relu', padding='same', kernel_initializer='normal')(conv6)

up7 = Conv2D(baseFilter*4, 2, activation='relu', padding='same', kernel_initializer='normal')(UpSampling2D(size=(2, 2))(conv6))
merge7 = concatenate([conv3, up7], axis=3)
conv7 = Conv2D(baseFilter*4, 3, activation='relu', padding='same', kernel_initializer='normal')(merge7)
conv7 = Conv2D(baseFilter*4, 3, activation='relu', padding='same', kernel_initializer='normal')(conv7)

up8 = Conv2D(baseFilter*2, 2, activation='relu', padding='same', kernel_initializer='normal')(UpSampling2D(size=(2, 2))(conv7))
merge8 = concatenate([conv2, up8], axis=3)
conv8 = Conv2D(baseFilter*2, 3, activation='relu', padding='same', kernel_initializer='normal')(merge8)
conv8 = Conv2D(baseFilter*2, 3, activation='relu', padding='same', kernel_initializer='normal')(conv8)

up9 = Conv2D(baseFilter, 2, activation='relu', padding='same', kernel_initializer='normal')(UpSampling2D(size=(2, 2))(conv8))
merge9 = concatenate([conv1, up9], axis=3)
conv9 = Conv2D(baseFilter, 3, activation='relu', padding='same', kernel_initializer='normal')(merge9)
conv9 = Conv2D(baseFilter, 3, activation='relu', padding='same', kernel_initializer='normal')(conv9)
conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='normal')(conv9)
conv10 = Conv2D(1, 1, activation="sigmoid")(conv9)

model = Model(input=inputs, output=conv10)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=300, verbose=1, mode='auto', restore_best_weights=True)

checkpointer = keras.callbacks.ModelCheckpoint("spot_segmenter_9000_2.sav", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

def lr_scheduler(epoch, lr):
    decay_rate = 0.95
    decay_step = 10
    if (epoch + 1) % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

def constant_lr_scheduler(epoch, lr):
    decay_rate = 5e-7
    decay_step = 10
    if (epoch + 1) % decay_step == 0 and epoch:
        return lr - decay_rate
    return lr

def scheduler_experimental(epoch, lr):
    lr = lr/((epoch+1)/2)
    print(lr)
    return lr

decay = keras.callbacks.LearningRateScheduler(constant_lr_scheduler, verbose=0)

model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=Adam(lr=1e-5), loss=generalized_dice_loss_w, metrics=['accuracy'])

#history = model.fit(x_train, y_train, batch_size=4, epochs=10000, validation_split=0.2, callbacks=[decay, early_stop, checkpointer])
#history = model.fit_generator(custom_generator(), epochs=10, validation_data=(x_val, y_val), steps_per_epoch=32, shuffle=True, callbacks=[checkpointer, early_stop, decay])
history = model.fit_generator(custom_generator_real(names), epochs=50, validation_data=(x_val, y_val), steps_per_epoch=len(names), shuffle=True, callbacks=[checkpointer, early_stop, decay])

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