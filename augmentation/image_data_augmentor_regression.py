import pandas as pd
import json
import cv2
import numpy as np
from augmentation.data_aug import *
from matplotlib import pyplot as plt

def reflect(value, centre):
    return value - 2 * (value - centre)

def get_flipped_images(image, eye, finn, maxX, maxY, minX, minY, visualize, name, path=None):

    maxX = int(maxX)
    maxY = int(maxY)
    minX = int(minX)
    minY = int(minY)

    imCentreX = int(image.shape[1] / 2)
    imCentreY = int(image.shape[0] / 2)

    reflectXMax = int(reflect(maxX, imCentreX))
    reflectXMin = int(minX - 2 * (minX - imCentreX))
    reflectYMax = int(maxY - 2 * (maxY - imCentreY))
    reflectYMin = int(minY - 2 * (minY - imCentreY))

    # Flipped in the X-Direction
    flip1 = cv2.flip(image, 1)

    x_train = np.array(flip1)
    eyeX = (int(reflect(eye[0], imCentreX)), int(eye[1]))
    finnX = (int(reflect(finn[0], imCentreX)), int(finn[1]))
    y_train = np.array([[reflectXMin, minY, reflectXMax, maxY], [eyeX[0], eyeX[1], finnX[0], finnX[1]]])
    if path == None:
        np.savez_compressed(("../Test/head_nparray/X_Rotated_{}").format(name), x=x_train, y=y_train)
    else:
        np.savez_compressed(("{}_X_Rotated").format(path), x=x_train, y=y_train)

    if visualize:
        cv2.rectangle(flip1, (reflectXMin, minY), (reflectXMax, maxY), (0, 255, 0), 5)
        cv2.circle(flip1, eyeX, 2, (255, 0, 0), 5)
        cv2.circle(flip1, finnX, 2, (0, 0, 255), 5)
        cv2.imshow("1", flip1)
    # ----------------------------------------------------------------------------------------------------

    # Flipped in the Y-Direction
    flip0 = cv2.flip(image, 0)

    x_train = np.array(flip0)
    eyeY = (int(eye[0]), int(reflect(eye[1], imCentreY)))
    finnY = (int(finn[0]), int(reflect(finn[1], imCentreY)))
    y_train = np.array([[minX, reflectYMin, maxX, reflectYMax], [eyeY[0], eyeY[1], finnY[0], finnY[1]]])
    if path == None:
        np.savez_compressed(("../Test/head_nparray/Y_Rotated_{}").format(name), x=x_train, y=y_train)
    else:
        np.savez_compressed(("{}_Y_Rotated").format(path), x=x_train, y=y_train)

    if visualize:
        cv2.rectangle(flip0, (minX, reflectYMin), (maxX, reflectYMax), (0, 255, 0), 5)
        cv2.circle(flip0, eyeY, 2, (255, 0, 0), 5)
        cv2.circle(flip0, finnY, 2, (0, 0, 255), 5)
        cv2.imshow("0", flip0)
    # ----------------------------------------------------------------------------------------------------

    # Flipped in the X AND Y-Direction
    flipN1 = cv2.flip(image, -1)

    x_train = np.array(flipN1)
    eyeXY = (int(reflect(eye[0], imCentreX)), int(reflect(eye[1], imCentreY)))
    finnXY = (int(reflect(finn[0], imCentreX)), int(reflect(finn[1], imCentreY)))
    y_train = np.array([[reflectXMin, reflectYMin, reflectXMax, reflectYMax], [eyeXY[0], eyeXY[1], finnXY[0], finnXY[1]]])
    if path == None:
        np.savez_compressed(("../Test/head_nparray/XY_Rotated_{}").format(name), x=x_train, y=y_train)
    else:
        np.savez_compressed(("{}_XY_Rotated").format(path), x=x_train, y=y_train)

    if visualize:
        cv2.rectangle(flipN1, (reflectXMin, reflectYMin), (reflectXMax, reflectYMax), (0, 255, 0), 5)
        cv2.circle(flipN1, eyeXY, 2, (255, 0, 0), 5)
        cv2.circle(flipN1, finnXY, 2, (0, 0, 255), 5)
        cv2.imshow("-1", flipN1)

    if visualize:
        cv2.imshow("original", image)
        cv2.waitKey(0)

# ALL CREDIT TO https://github.com/Paperspace/DataAugmentationForObjectDetection/tree/master/data_aug for augmentation

file = "../heads.csv"
labels = pd.read_csv(file)

print(labels.columns)

reductionMultiplier = 4  # --------------------------------------------------------------------------------------
visualize = True

for index, row in labels.iterrows():
    if index > 100000000:
        break
    print(index)
    string = row['Label']
    j = json.loads(string)
    eye = (int(j["Eye"][0]["geometry"]["x"] / reductionMultiplier), int(j["Eye"][0]["geometry"]["y"] / reductionMultiplier))

    finn = (int(j["Finn"][0]["geometry"]["x"] / reductionMultiplier), int(j["Finn"][0]["geometry"]["y"] / reductionMultiplier))

    hX = []
    hY = []
    for x in range(0, 4):
        hX.append(int(j["Head"][0]["geometry"][x]["x"] / reductionMultiplier))
        hY.append(int(j["Head"][0]["geometry"][x]["y"] / reductionMultiplier))

    maxX = np.amax(np.array(hX))
    maxY = np.amax(np.array(hY))

    minX = np.amin(np.array(hX))
    minY = np.amin(np.array(hY))

    id = row["External ID"]

    image = cv2.imread("../Test/Heads/" + id, 1)
    image = cv2.resize(image, (int(image.shape[1] / reductionMultiplier), int(image.shape[0] / reductionMultiplier)))
    image = image
    img = np.array(image).astype(np.uint8)
    y = np.array([eye[0], eye[1], finn[0], finn[1], minX, minY, maxX, maxY]).astype(float)

    eye_finn = [y[0], y[1], y[2], y[3]]
    bboxes = [[y[4], y[7], y[6], y[5]], eye_finn]
    bboxes = np.array(bboxes)

    if visualize:
        plotted_img = draw_rect(img, bboxes)
        plt.imshow(plotted_img)
        # Draw a point at the location (3, 9) with size 1000
        plt.scatter(bboxes[1][0], bboxes[1][1], s=20, c='red')
        plt.scatter(bboxes[1][2], bboxes[1][3], s=20)
        plt.show()
    else:
        # ---------- REGULAR IMAGE ----------
        ran = random.randint(0, 100)
        if ran < 20:
            np.savez_compressed(("../Test/head_val/{}").format(id), x=img, y=bboxes)
            get_flipped_images(img, (bboxes[1][0], bboxes[1][1]), (bboxes[1][2], bboxes[1][3]), bboxes[0][2], bboxes[0][1], bboxes[0][0], bboxes[0][3], False, ("regular_{}").format(id), path=("../Test/head_val/{}").format(id))
        elif ran > 98:
            np.savez_compressed(("../Test/head_val/{}").format(id), x=img, y=bboxes)
            get_flipped_images(img, (bboxes[1][0], bboxes[1][1]), (bboxes[1][2], bboxes[1][3]), bboxes[0][2], bboxes[0][1], bboxes[0][0], bboxes[0][3], False, ("regular_{}").format(id), path=("../Test/test_set/{}").format(id))
        else:
            np.savez_compressed(("../Test/head_nparray/{}").format(id), x=img, y=bboxes)
            get_flipped_images(img, (bboxes[1][0], bboxes[1][1]), (bboxes[1][2], bboxes[1][3]), bboxes[0][2], bboxes[0][1], bboxes[0][0], bboxes[0][3], False, ("regular_{}").format(id))

    # ---------- FLIPPED IMAGE ---------- TODO CURRENTLY BUGGED
    img_, bboxes_, flip = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
    if visualize and False:
        plotted_img = draw_rect(img_, bboxes_)
        plt.imshow(plotted_img)
        # Draw a point at the location (3, 9) with size 1000
        plt.scatter(bboxes_[1][2], bboxes_[1][1], s=20, c='red')
        plt.scatter(bboxes_[1][0], bboxes_[1][3], s=20)
        plt.show()

    # ---------- SCALE IMAGE ----------
    for i in range(0, 2):
        img_, bboxes_ = RandomScale(0.25, diff = True)(img.copy(), bboxes.copy())
        if visualize:
            plotted_img = draw_rect(img_, bboxes_)
            plt.imshow(plotted_img)
            # Draw a point at the location (3, 9) with size 1000
            plt.scatter(bboxes_[1][0], bboxes_[1][1], s=20, c='red')
            plt.scatter(bboxes_[1][2], bboxes_[1][3], s=20)
            plt.show()
        else:
            np.savez_compressed(("../Test/head_nparray/scaled_{}_{}").format(id, i), x=img_, y=bboxes_)
            get_flipped_images(img_, (bboxes_[1][0], bboxes_[1][1]), (bboxes_[1][2], bboxes_[1][3]), bboxes_[0][2], bboxes_[0][1], bboxes_[0][0], bboxes_[0][3], False, ("scaled_{}").format(id))

    # ---------- TRANSLATE IMAGE ----------
    img_, bboxes_ = RandomTranslate(0.1, diff = True)(img.copy(), bboxes.copy())
    if visualize:
        plotted_img = draw_rect(img_, bboxes_)
        plt.imshow(plotted_img)
        # Draw a point at the location (3, 9) with size 1000
        plt.scatter(bboxes_[1][0], bboxes_[1][1], s=20, c='red')
        plt.scatter(bboxes_[1][2], bboxes_[1][3], s=20)
        plt.show()
    else:
        np.savez_compressed(("../Test/head_nparray/translated_{}").format(id), x=img_, y=bboxes_)
        #get_flipped_images(img_, (bboxes_[1][0], bboxes_[1][1]), (bboxes_[1][2], bboxes_[1][3]), bboxes_[0][2], bboxes_[0][1], bboxes_[0][0], bboxes_[0][3], False, ("trans_{}").format(id))

    # ---------- ROTATE IMAGE ---------- TODO creates larger box due to rotation, this should be romved for eye and finn
    img_, bboxes_ = RandomRotate(20)(img.copy(), bboxes.copy())
    if visualize and False:
        plotted_img = draw_rect(img_, bboxes_)
        plt.imshow(plotted_img)
        # Draw a point at the location (3, 9) with size 1000
        plt.scatter(bboxes_[1][0], bboxes_[1][1], s=20, c='red')
        plt.scatter(bboxes_[1][2], bboxes_[1][3], s=20)
        plt.show()

    # ---------- SHEAR IMAGE ---------- TODO bugged when fish is right side, works when upside down
    img_, bboxes_ = RandomShear(0.2)(img.copy(), bboxes.copy())
    if visualize and False:
        plotted_img = draw_rect(img_, bboxes_)
        plt.imshow(plotted_img)
        # Draw a point at the location (3, 9) with size 1000
        plt.scatter(bboxes_[1][0], bboxes_[1][1], s=20, c='red')
        plt.scatter(bboxes_[1][2], bboxes_[1][3], s=20)
        plt.show()
    #else:
        #np.savez_compressed(("../Test/head_nparray/sheared_{}").format(id), x=img_, y=bboxes_)

    # ---------- HSV IMAGE ----------
    img_, bboxes_ = RandomHSV(100, 100, 100)(img.copy(), bboxes.copy())
    if visualize:
        plotted_img = draw_rect(img_, bboxes_)
        plt.imshow(plotted_img)
        # Draw a point at the location (3, 9) with size 1000
        plt.scatter(bboxes_[1][0], bboxes_[1][1], s=20, c='red')
        plt.scatter(bboxes_[1][2], bboxes_[1][3], s=20)
        plt.show()
    else:
        np.savez_compressed(("../Test/head_nparray/hsv_{}").format(id), x=img_, y=bboxes_)
        #get_flipped_images(img_, (bboxes_[1][0], bboxes_[1][1]), (bboxes_[1][2], bboxes_[1][3]), bboxes_[0][2], bboxes_[0][1], bboxes_[0][0], bboxes_[0][3], False, ("hsv_{}").format(id))

