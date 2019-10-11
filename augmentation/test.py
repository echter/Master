import numpy as np
import cv2
from augmentation.data_aug import *
from augmentation.bbox_utils import *
from augmentation.personal_utils import *

image = np.load("../Test/head_nparray/25427 070420.9-cam3-000108.png.npz")

img = image["x"].astype(np.uint8)
y = image["y"].astype(float)

# minX, minY, maxX, maxY
# y[4], y[5], y[6], y[7]

for i in range(0, 10):

    eye_finn = [y[0], y[1], y[2], y[3]]
    bboxes = [[y[4], y[7], y[6], y[5]],eye_finn]
    bboxes = np.array(bboxes)


    if False:
        plotted_img = draw_rect(img, bboxes)
        plt.imshow(plotted_img)
        # Draw a point at the location (3, 9) with size 1000
        plt.scatter(bboxes[1][0], bboxes[1][1], s=20, c='red')
        plt.scatter(bboxes[1][2], bboxes[1][3], s=20)
        plt.show()

    if False:
        img_, bboxes_, flip = RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
        plotted_img = draw_rect(img_, bboxes_)
        plt.imshow(plotted_img)
        # Draw a point at the location (3, 9) with size 1000
        plt.scatter(bboxes_[1][2], bboxes_[1][1], s=20, c='red')
        plt.scatter(bboxes_[1][0], bboxes_[1][3], s=20)
        plt.show()

    if False:
        img_, bboxes_ = RandomScale(0.2, diff = True)(img.copy(), bboxes.copy())
        plotted_img = draw_rect(img_, bboxes_)
        plt.imshow(plotted_img)
        plt.scatter(bboxes_[1][0], bboxes_[1][1], s=20, c='red')
        plt.scatter(bboxes_[1][2], bboxes_[1][3], s=20)
        plt.show()

    if False:
        img_, bboxes_ = RandomTranslate(0.3, diff = True)(img.copy(), bboxes.copy())
        plotted_img = draw_rect(img_, bboxes_)
        plt.imshow(plotted_img)
        plt.scatter(bboxes_[1][0], bboxes_[1][1], s=20, c='red')
        plt.scatter(bboxes_[1][2], bboxes_[1][3], s=20)
        plt.show()

    if False:
        img_, bboxes_ = RandomRotate(20)(img.copy(), bboxes.copy())
        plotted_img = draw_rect(img_, bboxes_)
        plt.imshow(plotted_img)
        plt.scatter(bboxes_[1][0], bboxes_[1][1], s=20, c='red')
        plt.scatter(bboxes_[1][2], bboxes_[1][3], s=20)
        plt.show()

    if False:
        img_, bboxes_ = RandomShear(0.2)(img.copy(), bboxes.copy())
        plotted_img = draw_rect(img_, bboxes_)
        plt.imshow(plotted_img)
        plt.scatter(bboxes_[1][0], bboxes_[1][1], s=20, c='red')
        plt.scatter(bboxes_[1][2], bboxes_[1][3], s=20)
        plt.show()

    if False:
        img_, bboxes_ = Resize(608)(img.copy(), bboxes.copy())
        plotted_img = draw_rect(img_, bboxes_)
        plt.imshow(plotted_img)
        plt.scatter(bboxes_[1][0], bboxes_[1][1], s=20, c='red')
        plt.scatter(bboxes_[1][2], bboxes_[1][3], s=20)
        plt.show()

    if False:
        img_, bboxes_ = RandomHSV(100, 100, 100)(img.copy(), bboxes.copy())
        plotted_img = draw_rect(img_, bboxes_)
        plt.imshow(plotted_img)
        plt.scatter(bboxes_[1][0], bboxes_[1][1], s=20, c='red')
        plt.scatter(bboxes_[1][2], bboxes_[1][3], s=20)
        plt.show()