import json

import cv2
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Input, Dropout, concatenate, Dense, ZeroPadding2D, Convolution2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from scipy.spatial import distance
from skimage.measure import compare_ssim as ssim
import os
import argparse

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

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

import keras.losses
keras.losses.generalized_dice_loss_w = generalized_dice_loss_w

eye_finn_net = keras.models.load_model("eye_finn_9000_mse1783.sav")
head_net = keras.models.load_model("head_bb_finder_2136.sav")
segmenter = keras.models.load_model("dice022264.sav")

counter = 0

ef_list = []
image_list = []

for filename in os.listdir("Test/head_nparray"):
    if counter > 1000:
        break

    print(filename)

    file = np.load("Test/head_nparray/" + filename)["x"]
    pred = head_net.predict(file.reshape(1, 500, 1368, 3))[0]
    pred = [int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])]

    headXYMin = (pred[0], pred[1])
    headXYMax = (pred[2], pred[3])
    box = cv2.rectangle(file.copy(), headXYMin, headXYMax, (0, 255, 0), 3)

    roi = file[pred[3]:pred[3]+(pred[1]-pred[3]), pred[2]+(pred[0]-pred[2]):pred[2]]

    head = cv2.resize(roi, (512, 512))
    seg_pred = segmenter.predict(head.reshape(1, 512, 512, 3))

    pred = eye_finn_net.predict(file.reshape(1, 500, 1368, 3))[0]
    ef_pred = [int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3])]

    eye = (int(ef_pred[0]), int(ef_pred[1]))
    finn = (int(ef_pred[2]), int(ef_pred[3]))
    cv2.circle(file, eye, 4, (0, 0, 255), 2)
    cv2.circle(file, finn, 4, (255, 0, 0), 2)
    cv2.line(file, eye, finn, (0, 255, 0), 10)

    ef_list.append(ef_pred)
    image_list.append(seg_pred.reshape(512, 512) * 255)
    #image_list.append(file)

    counter = counter + 1

def mse(original, potential_match):

    match_points = 0

    for i in range(original.shape[0]):
        for j in range(original.shape[1]):
            if original[i][j] > 0:
                if potential_match[i][j] > 0:
                    match_points+=1

    return match_points

alpha_slider_max = 100
title_window = 'Linear Blend'

def match_images(original_image, original_ef, images, images_ef):
    score_list = []
    index = 0
    for image in images:
        rotation = angle_between(original_ef, images_ef[index])
        M = cv2.getRotationMatrix2D((original_image.shape[1] / 2, original_image.shape[0] / 2), rotation, 1)
        result_image = cv2.warpAffine(image, M, (original_image.shape[1], original_image.shape[0]))

        result_image[result_image > 1] = 255
        result_image[result_image <= 1] = 0

        cv2.imwrite(("Test/head_network_results/segmentations/{}.png").format(), image)


        score = mse(original_image, result_image)

        if score > 800:
            print(images_ef[index])
            print(original_ef)
            print(rotation)
            print(("Score for image {}:      {}").format(index, score))
            #cv2.imwrite(("Test/head_network_results/test/{}_{}.png").format(index, score), image)
            #plt.imshow(result_image + original_image)
            #plt.show()

        score_list.append(score)

        index = index + 1

    return score_list

scores = match_images(image_list[0], ef_list[0], image_list, ef_list)

plt.plot(scores)
plt.show()
