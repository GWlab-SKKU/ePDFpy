import file
import numpy as np
import cv2
import json
import inspect
import os
from pathlib import Path
import pandas as pd

settings = json.load(open("settings.json"))


def get_sample_img():
    return file.load_mrc_img("./assets/Camera 230 mm Ceta 20210312 1333_50s_20f_area01.mrc")

def get_mask_data():
    return np.loadtxt('./assets/mask_data.txt',delimiter=',').astype(np.uint8)

def get_sample_azimuthal_average():
    return np.loadtxt('./assets/sample_azavg.csv',delimiter=',')

def create_estimated_mask(center=None,radius=None):
    raw, img = get_sample_img()
    x1 = 900
    x2 = 1250
    y1 = 900
    # plt.imshow(img[x1:x2,y1:])
    img_slice = img[x1:x2,y1:]
    thresh = cv2.inRange(img_slice,0,130)
    thresh[0:100,700:]=0
    thresh[240:,700:]=0
    mask = np.zeros(img.shape)
    mask[x1:x2,y1:] = thresh

    kernel = np.ones((50, 50), np.uint8)
    mask = cv2.dilate(mask, kernel, 1)

    if center is not None:
        c_x,c_y=center
        c_x=np.uint16(c_x)
        c_y=np.uint16(c_y)

        if radius is not None:
            mask = cv2.circle(mask, (c_x, c_y), radius, 255, -1)
        else:
            mask = cv2.circle(mask, (c_x, c_y), 100, 255, -1)

    mask = np.uint8(mask)
    return mask


def save_settings(settings_to_save):
    json.dump(settings_to_save, open("settings.json", 'w'), indent=2)

lst_atomic_number_symbol = None
def get_atomic_number_symbol():
    global lst_atomic_number_symbol
    if lst_atomic_number_symbol is None:
        df_atomic_number_symbol = pd.read_csv("assets/Parameter_files/atomicNumber-symbol.csv")
        lst_atomic_number_symbol = df_atomic_number_symbol['title'].tolist()
        lst_atomic_number_symbol.insert(0,"None")
    return lst_atomic_number_symbol

np_kirkland = None
def get_kirkland_2010():
    global np_kirkland
    if np_kirkland is None:
        np_kirkland = np.loadtxt("assets/Parameter_files/Kirkland_2010.txt")
    return np_kirkland



if __name__ == '__main__':
    # mask = create_estimated_mask()
    # np.savetxt('./assets/mask_data.txt',mask,fmt='%i',delimiter=',')

    # print(settings['show_center_line']==True)
    print(get_atomic_number_symbol())

