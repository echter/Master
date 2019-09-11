import pandas as pd
import json
from PIL import Image
import requests
from io import BytesIO
from matplotlib import pyplot as plt
from urllib.request import urlopen
from PIL import Image
import cv2
import numpy as np

image = np.load("Test/val/cropped_v5.png.npz")["x"]
plt.imshow(image)
plt.show()

file = "test/label/labels.csv"
labels = pd.read_csv(file)

print(labels.columns)

for index, row in labels.iterrows():
    id = row["External ID"]
    string = row['Label']
    j = json.loads(string)
    print(j)
    objects = j["objects"]

    test = True

    for object in objects:
        if object["title"] != "Eye":
            url = object["instanceURI"]
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            pic = np.array(img)
            pic = pic[:,:,0]

            if test:
                image = pic
                test = False
            else:
                image = cv2.add(image, pic)

    original = cv2.imread("Test/Data/" + id)

    image = cv2.resize(image, (512, 512))
    original = cv2.resize(original, (512, 512))

    image = np.array(image)

    mask = image > 1
    image = mask.astype(int)

    x_train = original
    y_train = image

    np.savez_compressed(("Test/nparray/{}").format(id), x=x_train, y=y_train)

    unique, counts = np.unique(image, return_counts=True)
    print(unique)
    print(counts)