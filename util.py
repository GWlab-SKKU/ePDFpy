import file
from pathlib import Path
import numpy as np
import cv2
import json
import inspect
import os
from pathlib import Path
import pandas as pd
from PIL import Image

setting_path = "settings/default.json"

def get_sample_img():
    return file.load_mrc_img("./assets/Camera 230 mm Ceta 20210312 1333_50s_20f_area01.mrc")


# def get_mask_data():
#     return np.loadtxt('./assets/mask_data.txt',delimiter=',').astype(np.uint8)

def get_sample_azimuthal_average():
    return np.loadtxt('./assets/sample_azavg.csv', delimiter=',')


def create_estimated_mask(center=None, radius=None, kernel_size=50):
    raw, img = get_sample_img()
    x1 = 900
    x2 = 1250
    y1 = 900
    # plt.imshow(img[x1:x2,y1:])
    img_slice = img[x1:x2, y1:]
    thresh = cv2.inRange(img_slice, 0, 130)
    thresh[0:100, 700:] = 0
    thresh[240:, 700:] = 0
    mask = np.zeros(img.shape)
    mask[x1:x2, y1:] = thresh

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel, 1)

    if center is not None:
        c_x, c_y = center
        c_x = np.uint16(c_x)
        c_y = np.uint16(c_y)

        if radius is not None:
            mask = cv2.circle(mask, (c_x, c_y), radius, 255, -1)
        else:
            mask = cv2.circle(mask, (c_x, c_y), 100, 255, -1)

    mask = np.uint8(mask)
    return mask


def save_settings(settings_to_save):
    json.dump(settings_to_save, open("settings/default.json", 'w'), indent=2)


lst_atomic_number_symbol = None


def get_atomic_number_symbol():
    global lst_atomic_number_symbol
    if lst_atomic_number_symbol is None:
        df_atomic_number_symbol = pd.read_csv("assets/Parameter_files/atomicNumber-symbol.csv")
        lst_atomic_number_symbol = df_atomic_number_symbol['title'].tolist()
        lst_atomic_number_symbol.insert(0, "None")
    return lst_atomic_number_symbol


np_kirkland = None


def get_kirkland_2010():
    global np_kirkland
    if np_kirkland is None:
        np_kirkland = np.loadtxt("assets/Parameter_files/scattering_factor/Kirkland_2010.txt")
    return np_kirkland


def load_previous_dc_azavg(dc):
    current_folder, current_file_full_name = os.path.split(dc.mrc_file_path)
    current_file_name, current_ext = os.path.splitext(current_file_full_name)
    analysis_folder = os.path.join(current_folder, file.analysis_folder_name)  # todo: temp

    load_save = os.path.join(analysis_folder, current_file_name + " azav")

    i_slice = [default_setting.intensity_range_1, default_setting.intensity_range_2, default_setting.slice_count]
    if i_slice:
        load_save = load_save + " center" + str(i_slice[0]) + "to" + str(i_slice[1]) + "_" + str(i_slice[2])
        # load_save = load_save + " center" + str(110) + "to" + str(120) + "_" + str(1)

    # add extension
    load_save = load_save + ".txt"

    if not os.path.isfile(load_save):
        print('There is no such file:', load_save)
        return
    dc.azavg = np.loadtxt(load_save)
    return dc.azavg


def load_previous_tiff(dc):
    current_folder, current_file_full_name = os.path.split(dc.mrc_file_path)
    current_file_name, current_ext = os.path.splitext(current_file_full_name)
    analysis_folder = os.path.join(current_folder, file.analysis_folder_name)  # todo: temp

    load_save = os.path.join(analysis_folder, current_file_name + "_img.tiff")

    if not os.path.isfile(load_save):
        print('There is no such file:', load_save)
        return
    dc.display_img = np.array(Image.open(load_save))
    return dc.display_img


def get_multiple_dc(folder_path):
    pass


class DefaultSetting:
    def __init__(self):
        self.intensity_range_1 = None
        self.intensity_range_2 = None
        self.slice_count = None
        self.show_center_line = None
        self.show_beam_stopper_mask = None

        self.calibration_factor = None
        self.calibration_factor_step = None
        self.fit_at_q_step = None
        self.N_step = None
        self.dr_step = None
        self.damping = None
        self.damping_step = None
        self.rmax = None
        self.rmax_step = None
        self.load()

    def load(self):
        if os.path.isfile(setting_path):
            dic_ = json.load(open(setting_path))
            for key, value in dic_.items():
                setattr(self, key, value)
        else:
            Path(setting_path).touch()

    def save_settings(self):
        json.dump(vars(self), open(setting_path, 'w'), indent=2)




default_setting = DefaultSetting()

if __name__ == '__main__':
    # mask = create_estimated_mask()
    # np.savetxt('./assets/mask_data.txt',mask,fmt='%i',delimiter=',')

    # print(settings['show_center_line']==True)
    print(get_atomic_number_symbol())
