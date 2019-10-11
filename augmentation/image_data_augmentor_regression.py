import pandas as pd
import json
import cv2
import numpy as np
from augmentation.bbox_utils import *
from matplotlib import pyplot as plt

# ALL CREDIT TO https://github.com/Paperspace/DataAugmentationForObjectDetection/tree/master/data_aug for augmentation

file = "../heads.csv"
labels = pd.read_csv(file)

print(labels.columns)

reductionMultiplier = 8  # --------------------------------------------------------------------------------------

for index, row in labels.iterrows():
    if index > 0:
        break
    print(index)
    string = row['Label']
    j = json.loads(string)

    hX = []
    hY = []
    for x in range(0, 4):
        hX.append(int(j["Head"][0]["geometry"][x]["x"] / reductionMultiplier))
        hY.append(int(j["Head"][0]["geometry"][x]["y"] / reductionMultiplier))

    maxX = np.amax(np.array(hX))
    maxY = np.amax(np.array(hY))

    minX = np.amin(np.array(hX))
    minY = np.amin(np.array(hY))

    eye = (int(j["Eye"][0]["geometry"]["x"] / reductionMultiplier), int(j["Eye"][0]["geometry"]["y"] / reductionMultiplier))
    finn = (int(j["Finn"][0]["geometry"]["x"] / reductionMultiplier), int(j["Finn"][0]["geometry"]["y"] / reductionMultiplier))

    bbHead = [minX, maxY, maxX, minY]
    bbEyeFinn = [eye[0], eye[1], finn[0], finn[1]]
    bboxes = [bbHead, bbEyeFinn]

    id = row["External ID"]

    img = cv2.imread("Test/Heads/" + id)
    img = cv2.resize(img, (int(img.shape[1] / reductionMultiplier), int(img.shape[0] / reductionMultiplier)))

    plotted_img = draw_rect(img, bboxes)
    plt.imshow(plotted_img)
    # Draw a point at the location (3, 9) with size 1000
    plt.scatter(bboxes[1][0], bboxes[1][1], s=20, c='red')
    plt.scatter(bboxes[1][2], bboxes[1][3], s=20)
    plt.show()

