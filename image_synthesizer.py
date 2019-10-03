import cv2
import random
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

for i in range(0, 0):
    s_img = cv2.imread("Test/Data/cropped_v4.png", 0)
    s_img = cv2.resize(s_img, (int(s_img.shape[1] / 2), int(s_img.shape[0] / 2)))
    og = cv2.resize(s_img, (int(s_img.shape[1] / 2), int(s_img.shape[0] / 2)))
    rows = s_img.shape[0]
    cols = s_img.shape[1]

    # TODO ADD NOISE
    noise = np.zeros((rows, cols))
    m = (10, 12, 34)
    sigma = (1, 5, 50)
    cv2.randn(noise, m, sigma)

    cv2.imshow("f", s_img)
    s_img += noise

    cv2.imshow("dst", s_img)
    cv2.waitKey(0)


for i in range(0, 20):

    l_img = cv2.imread("spotless.png", 1)
    y_mask = np.zeros((l_img.shape[0], l_img.shape[1]))

    for i in range(0, random.randint(4, 13)):

        s_img = cv2.imread("dot2.png", -1)

        s_img[:, :, 0:2] = s_img[:, :, 0:2] * (random.randint(1, 100) / 70)

        reductionMultiplierX = (random.randint(1,80) / 40) + 1
        reductionMultiplierY = (random.randint(1,80) / 40) + 1

        s_img = cv2.resize(s_img, (int(s_img.shape[1] / reductionMultiplierX), int(s_img.shape[0] / reductionMultiplierY)))

        rows = cols = s_img.shape[0]

        #M = np.float32([[1, 0, 100], [0, 1, 50]])
        #dst = cv2.warpAffine(s_img, M, (cols, rows))

        # TODO ---------------- FIX THE CUTOFF

        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random.randint(0, 360), 0.4)
        s_img = cv2.warpAffine(s_img, M, (cols , rows))

        s_img = cv2.resize(s_img, (int(s_img.shape[1] * 2.5), int(s_img.shape[0] * 2.5)))

        #cv2.imshow("partial", s_img)
        #cv2.waitKey(0)

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

        #cv2.imshow("why", s_img)
        #cv2.waitKey(0)
        #print(s_img.shape)

        s_img = s_img.reshape(s_img.shape[1], s_img.shape[0], 1)

        if l_cut.shape[0] == l_cut.shape[1] == s_img.shape[0] == s_img.shape[1] == y_mask_cut.shape[0] == y_mask_cut.shape[1]:

            simg = alpha_s * s_img[:, :, 0]

            y_mask[y1:y2, x1:x2] += mask

            for c in range(0, 3):
                roi = alpha_l * l_cut[:, :, c]
                l_img[y1:y2, x1:x2, c] = (simg + roi)

    cv2.imshow("result", l_img)
    cv2.imshow("mask", y_mask)
    cv2.imwrite(("Showcase/{}.jpg").format(i), l_img)
    cv2.waitKey(0)