from datacube import DataCube

import mrcfile
import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog
import json
import copy
import pandas as pd
from pathlib import Path
import definitions
import re

ePDFpy_analysis_folder_name = "Analysis ePDFpy"
preset_ext = ".preset.json"
azavg_ext = ".azavg.txt"
normstd_ext = ".normstd.txt"
data_q_ext = ".q.csv"
data_r_ext = ".r.csv"
image_ext = ".img.png"
rdf_screen_ext = ".rdf.png"

def load_mrc_img(fp):
    with mrcfile.open(fp) as mrc:
        raw_img = mrc.data
    easy_img = np.log(np.abs(raw_img) + 1)
    easy_img = easy_img / easy_img.max() * 255
    easy_img = easy_img.astype('uint8')
    return raw_img, easy_img


def get_file_list_from_path(fp, extension=None):
    files = Path(fp).rglob("*" + extension)
    file_list = []
    for _file in files:
        file_list.append(str(_file.absolute()))
    file_list.sort()
    return file_list


def make_analyse_folder(datacube):
    dc_filepath = datacube.load_file_path
    if os.path.isdir(dc_filepath):
        ePDFpy_analysis_folder = os.path.join(dc_filepath, ePDFpy_analysis_folder_name)
    else:
        current_folder, current_file_full_name = os.path.split(dc_filepath)
        ePDFpy_analysis_folder = os.path.join(current_folder, ePDFpy_analysis_folder_name)

    if not os.path.isdir(ePDFpy_analysis_folder):
        try:
            os.makedirs(ePDFpy_analysis_folder)
        except:
            print('Failed to make directory:', ePDFpy_analysis_folder)
            return False

    current_file_name = os.path.splitext(current_file_full_name)[0]
    if datacube.data_quality is not None:
        current_file_name = "({}){}".format(datacube.data_quality, current_file_name)
    final_analysis_folder = os.path.join(ePDFpy_analysis_folder, current_file_name)
    if not os.path.isdir(final_analysis_folder):
        try:
            os.makedirs(final_analysis_folder)
            return final_analysis_folder
        except:
            print('Failed to make directory:', final_analysis_folder)
            return False

    return final_analysis_folder


def save_preset_default(datacube, main_window):
    imgPanel = main_window.profile_extraction.img_panel
    # Types of datacube source : azavg, mrc, None, preset&azavg, preset&mrc, preset&None

    if datacube.azavg_file_path is not None:
        # when you load "azavg only" or "preset that based on azavg".
        current_folder_path, file_name = os.path.split(datacube.azavg_file_path)
        file_short_name, file_ext = os.path.splitext(file_name)
        analysis_folder_path = current_folder_path
    if datacube.mrc_file_path is None and datacube.azavg_file_path is None:
        # When there is no source ( when load from other program )
        fp, _ = QFileDialog.getSaveFileName(filter="preset Files (*.preset.json)")
        current_folder_path, file_name = os.path.split(fp)
        file_short_name = str(file_name)[:str(file_name).find(preset_ext)]
        analysis_folder_path = current_folder_path
    if datacube.mrc_file_path is not None:
        # When you load img files
        if datacube.preset_file_path is None:
            current_folder_path, file_name = os.path.split(datacube.mrc_file_path)
            file_short_name, file_ext = os.path.splitext(file_name)
            analysis_folder_path = os.path.join(current_folder_path,
                                                ePDFpy_analysis_folder_name,
                                                "({}){}".format(datacube.data_quality, file_short_name))
        else:
            # with preset
            current_folder_path, file_name = os.path.split(datacube.preset_file_path)
            file_short_name = str(file_name)[:str(file_name).find(preset_ext)]
            # data quality folder name
            new_folder_path = re.sub(r'\(L.\)',"({})".format(datacube.data_quality), current_folder_path)
            new_folder_path = re.sub(r'\(None\)',"({})".format(datacube.data_quality), new_folder_path)
            if current_folder_path != new_folder_path:
                os.rename(current_folder_path, new_folder_path)
            analysis_folder_path = new_folder_path


    os.makedirs(analysis_folder_path, exist_ok=True)

    preset_path = os.path.join(analysis_folder_path, file_short_name + preset_ext)
    azavg_path = os.path.join(analysis_folder_path, file_short_name + azavg_ext)
    normstd_path = os.path.join(analysis_folder_path, file_short_name + normstd_ext)
    data_q_path = os.path.join(analysis_folder_path, file_short_name + data_q_ext)
    data_r_path = os.path.join(analysis_folder_path, file_short_name + data_r_ext)
    img_path = os.path.join(analysis_folder_path, file_short_name + image_ext)
    rdf_screen_path = os.path.join(analysis_folder_path, file_short_name + rdf_screen_ext)
    datacube.preset_file_path = preset_path

    # save azavg
    if datacube.azavg is not None:
        np.savetxt(azavg_path, datacube.azavg)
    # save normalized std
    if datacube.azvar is not None:
        np.savetxt(normstd_path, datacube.azvar)
    # save q data
    if datacube.q is not None:
        lst = ['q','Iq','Autofit','phiq','phiq_damp']
        df = pd.DataFrame({name:getattr(datacube, name) for name in lst if getattr(datacube, name) is not None})
        df.to_csv(data_q_path, index=None)
    # save r data
    if datacube.r is not None:
        lst = ['r','Gr']
        df = pd.DataFrame({name:getattr(datacube, name) for name in lst if getattr(datacube, name) is not None})
        df.to_csv(data_r_path, index=None)
    # save img data
    if imgPanel is not None and datacube.img is not None:
        imgPanel.imageView.export(img_path)
    # save screenshot
    if datacube.Gr is not None:
        main_window.PDF_analyser.grab().save(rdf_screen_path)

    ################ save preset #################
    presets = vars(copy.copy(datacube))

    # convert to relative path
    if presets['mrc_file_path'] is not None:
        presets['mrc_file_path'] = os.path.relpath(datacube.mrc_file_path, os.path.split(preset_path)[0])
        presets['mrc_file_path'] = presets['mrc_file_path'].replace('\\', '/')  # compatibility between windows and linux

    # remove data that not support to save as json
    for key, value in dict(presets).items():
        if type(value) not in [int, str, float, list, np.float64, np.int64]:
            del presets[key]

    # Don't save in preset file
    remove_list = ['load_file_path','preset_file_path']
    for remove_item in remove_list:
        if remove_item in dict(presets).keys():
            del presets[remove_item]

    # int64 exception handling
    for key, value in presets.items():
        if type(value) == np.int64:
            presets[key] = int(value)
        if type(value) == np.float64:
            presets[key] = float(value)

    if presets['center'][0] is not None:
        presets['center'] = [int(presets['center'][0]), int(presets['center'][1])]

    print("save data:", presets)
    print(preset_path)
    json.dump(presets, open(preset_path, 'w'), indent=2)
    return True

def load_blank_img():
    fp, _ = QFileDialog.getOpenFileName(filter="mrc (*.mrc)")
    if fp == '':
        return
    with mrcfile.open(fp) as mrc:
        raw_img = mrc.data
    return raw_img

def load_preset(fp:str=None, dc:DataCube=None) -> DataCube:
    if fp is None:
        fp, _ = QFileDialog.getOpenFileName(filter="preset Files (*.preset.json)")
    if fp == '':
        return

    if dc is None:
        dc = DataCube()
    content = json.load(open(fp))

    azavg_path = os.path.join(os.path.split(fp)[0], fp[:fp.rfind(preset_ext)] + azavg_ext)
    data_r_path = os.path.join(os.path.split(fp)[0], fp[:fp.rfind(preset_ext)] + data_r_ext)
    data_q_path = os.path.join(os.path.split(fp)[0], fp[:fp.rfind(preset_ext)] + data_q_ext)
    normstd_path = os.path.join(os.path.split(fp)[0], fp[:fp.rfind(preset_ext)] + normstd_ext)

    dc.load_file_path = fp
    dc.preset_file_path = fp
    if os.path.isfile(azavg_path):
        df_azavg = np.loadtxt(azavg_path)
        dc.azavg = df_azavg
    if os.path.isfile(normstd_path):
        df_normstd = np.loadtxt(normstd_path)
        dc.azvar = df_normstd
    if os.path.isfile(data_r_path):
        df_r = pd.read_csv(data_r_path)
        for column in df_r.columns:
            setattr(dc, column, df_r[column].to_numpy())
    if os.path.isfile(data_q_path):
        df_q = pd.read_csv(data_q_path)
        for column in df_q.columns:
            setattr(dc, column, df_q[column].to_numpy())

    # convert relative path to absolute path
    if 'mrc_file_path' in content.keys():
        content['mrc_file_path'] = os.path.abspath(os.path.join(fp, "..", content['mrc_file_path']))
        content['mrc_file_path'] = os.path.abspath(os.path.join(fp, "..", content['mrc_file_path']))

    # put content in DataCube
    for key, value in content.items():
        if key in vars(dc).keys():
            setattr(dc, key, value)

    return dc


def save_pdf_setting_manual(dc_file_path):
    fp, _ = QFileDialog.getSaveFileName()
    json.dump(fp, open(fp, 'w'), indent=2)
    return True


def load_element_preset():
    if not os.path.isfile(definitions.ELEMENT_PRESETS_PATH):
        element_preset = [None]*5
        json.dump(element_preset, open(definitions.ELEMENT_PRESETS_PATH, 'w'), indent=2)
        return element_preset
    return json.load(open(definitions.ELEMENT_PRESETS_PATH))


def save_element_preset(data):
    json.dump(data, open(definitions.ELEMENT_PRESETS_PATH,'w'), indent=2)

# def load_azavg_from_preset(preset_path:str):
#     preset_path.rfind(preset_ext)


def load_azavg(fp) -> np.ndarray:
    current_folder_path, file_name = os.path.split(fp)
    file_short_name, file_ext = os.path.splitext(file_name)
    if file_ext == ".csv":
        return np.loadtxt(fp, delimiter=",")
    if file_ext == ".txt":
        return np.loadtxt(fp)


def save_azavg_only(azavg):
    fp, ext = QFileDialog.getSaveFileName(filter="csv (*.csv);; txt (*.txt)")
    if fp is '':
        return
    if 'csv' in ext:
        np.savetxt(fp,azavg,delimiter=',')
    if 'txt' in ext:
        np.savetxt(fp,azavg)




if __name__ == '__main__':
    # file_list = get_file_list_from_path('/mnt/experiment/TEM diffraction/210312','.mrc')
    # print(os.path.split(file_list[0]))
    # print(get_file_list_from_path('/mnt/experiment/TEM diffraction/210312','.mrc'))
    # pth="/mnt/experiment/TEM diffraction/210215/sample47_TiGe44_bot_AD/Camera 230 mm Ceta 20210215 1438_2s_1f_area01.mrc"
    # print(os.path.split(pth))
    from PyQt5.QtWidgets import QMainWindow
    from PyQt5 import QtWidgets

    qtapp = QtWidgets.QApplication([])
    # load_azavg_manual(r"Y:\experiment\TEM diffraction\210520\Analysis pdf_tools\Camera 230 mm Ceta 20210520 1306_Au azav center110to120_1.txt")
    # load_azavg_manual()
    qtapp.exec_()
    pass
