import pandas as pd
import json
import cv2
import numpy as np

def reflect(value, centre):
    return value - 2 * (value - centre)

def get_flipped_images(image, eye, finn, maxX, maxY, minX, minY, visualize):

    imCentreX = int(image.shape[1] / 2)
    imCentreY = int(image.shape[0] / 2)

    reflectXMax = reflect(maxX, imCentreX)
    reflectXMin = minX - 2 * (minX - imCentreX)
    reflectYMax = maxY - 2 * (maxY - imCentreY)
    reflectYMin = minY - 2 * (minY - imCentreY)

    # Flipped in the X-Direction
    flip1 = cv2.flip(image, 1)

    x_train = np.array(flip1)
    eyeX = (reflect(eye[0], imCentreX), eye[1])
    finnX = (reflect(finn[0], imCentreX), finn[1])
    y_train = np.array([eyeX[0], eyeX[1], finnX[0], finnX[1], reflectXMin, minY, reflectXMax, maxY])
    np.savez_compressed(("Test/head_nparray/X_Rotated_{}").format(id), x=x_train, y=y_train)

    if visualize:
        cv2.rectangle(flip1, (reflectXMin, minY), (reflectXMax, maxY), (0, 255, 0), 5)
        cv2.circle(flip1, eyeX, 2, (255, 0, 0), 5)
        cv2.circle(flip1, finnX, 2, (0, 0, 255), 5)
        cv2.imshow("1", flip1)
    # ----------------------------------------------------------------------------------------------------

    # Flipped in the Y-Direction
    flip0 = cv2.flip(image, 0)

    x_train = np.array(flip0)
    eyeY = (eye[0], reflect(eye[1], imCentreY))
    finnY = (finn[0], reflect(finn[1], imCentreY))
    y_train = np.array([eyeY[0], eyeY[1], finnY[0], finnY[1], minX, reflectYMin, maxX, reflectYMax])
    np.savez_compressed(("Test/head_nparray/Y_Rotated_{}").format(id), x=x_train, y=y_train)

    if visualize:
        cv2.rectangle(flip0, (minX, reflectYMin), (maxX, reflectYMax), (0, 255, 0), 5)
        cv2.circle(flip0, eyeY, 2, (255, 0, 0), 5)
        cv2.circle(flip0, finnY, 2, (0, 0, 255), 5)
        cv2.imshow("0", flip0)
    # ----------------------------------------------------------------------------------------------------

    # Flipped in the X AND Y-Direction
    flipN1 = cv2.flip(image, -1)

    x_train = np.array(flipN1)
    eyeXY = (reflect(eye[0], imCentreX), reflect(eye[1], imCentreY))
    finnXY = (reflect(finn[0], imCentreX), reflect(finn[1], imCentreY))
    y_train = np.array([eyeXY[0], eyeXY[1], finnXY[0], finnXY[1], reflectXMin, reflectYMin, reflectXMax, reflectYMax])
    np.savez_compressed(("Test/head_nparray/XY_Rotated_{}").format(id), x=x_train, y=y_train)

    if visualize:
        cv2.rectangle(flipN1, (reflectXMin, reflectYMin), (reflectXMax, reflectYMax), (0, 255, 0), 5)
        cv2.circle(flipN1, eyeXY, 2, (255, 0, 0), 5)
        cv2.circle(flipN1, finnXY, 2, (0, 0, 255), 5)
        cv2.imshow("-1", flipN1)

    if visualize:
        cv2.imshow("original", image)
        cv2.waitKey(0)

file = "heads.csv"
labels = pd.read_csv(file)

print(labels.columns)

reductionMultiplier = 8  # --------------------------------------------------------------------------------------

for index, row in labels.iterrows():
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

    image = cv2.imread("Heads/" + id)
    image = cv2.resize(image, (int(image.shape[1] / reductionMultiplier), int(image.shape[0] / reductionMultiplier)))

    get_flipped_images(image, eye, finn, maxX, maxY, minX, minY, False)

    #cv2.rectangle(image, (minX, minY), (maxX, maxY), (0, 255, 0), 1)
    #cv2.circle(image, eye, 2, (255, 0, 0), 1)
    #cv2.circle(image, finn, 2, (0, 0, 255), 1)
    #cv2.imshow("lmao", image)
    #cv2.waitKey(0)

    x_train = np.array(image)
    y_train = np.array([eye[0], eye[1], finn[0], finn[1], minX, minY, maxX, maxY])

    np.savez_compressed(("Test/head_nparray/{}").format(id), x=x_train, y=y_train)

    if index > 5000:
        break

    print(index)