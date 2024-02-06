import cv2
import numpy as np
import cv2
import sys
import mrcfile
from scipy import signal
import pandas as pd

def find_polygon(raw_img):
    kernlen = 19
    std = 4
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    # pd.DataFrame(gkern2d)

    img = np.abs(raw_img) + 1
    log_img = np.log(img)
    # blur = cv2.GaussianBlur(log_img,(19,19),0)  # Smoothing
    blur = cv2.filter2D(log_img, -1, gkern2d)
    blur.max()

    log_img = np.log(img)
    blur = cv2.GaussianBlur(log_img, (19, 19), 0)
    sobel1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel2 = sobel1.transpose()
    sobel_x = cv2.filter2D(src=blur, ddepth=-1, kernel=sobel1)
    sobel_y = cv2.filter2D(src=blur, ddepth=-1, kernel=sobel2)
    sobel_magnitude = sobel_x ** 2 + sobel_y ** 2

    thresh = cv2.threshold(cv2.convertScaleAbs(sobel_magnitude), 0, 1, cv2.THRESH_BINARY)[1]  # Make binary images
    # plt.imshow(thresh)
    # plt.show()
    kernel = np.ones(np.round(np.array(img.shape)/300).astype(np.uint8), np.uint8)
    trimed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel=kernel)  # Remove noise
    # plt.imshow(trimed_thresh)
    # plt.show()
    if np.sum(trimed_thresh) == 0:
        print("failed to find contours")
        return

    ######## Find Center #########
    x_size = trimed_thresh.shape[0]
    y_size = trimed_thresh.shape[1]
    y_grid, x_grid = np.meshgrid(np.arange(x_size), np.arange(y_size))
    x_center = int(np.sum(x_grid * trimed_thresh) / np.sum(trimed_thresh == 1))
    y_center = int(np.sum(y_grid * trimed_thresh) / np.sum(trimed_thresh == 1))

    print(x_center, y_center)

    ######## Fill Mask #########
    floodfill_img = trimed_thresh.copy()
    cv2.floodFill(floodfill_img, None, (y_center, x_center), 1)
    floodfill_img = cv2.morphologyEx(floodfill_img, cv2.MORPH_DILATE, kernel=np.ones((3,3), np.uint8))
    # plt.imshow(floodfill_img)
    # plt.show()

    contours, hierarchy = cv2.findContours(image=floodfill_img, mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_TC89_KCOS)
    # image_copy = floodfill_img.copy()
    # a = cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=20,
    #                      lineType=cv2.LINE_AA)

    if len(contours) != 1:
        print("failed to find contours")
        return

    epsilon1 = 0.005 * cv2.arcLength(contours[0], True)
    approx1 = cv2.approxPolyDP(contours[0], epsilon1, True)
    return approx1