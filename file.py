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

analysis_folder_name = "Analysis ePDFpy"
preset_ext = ".preset.json"
azavg_ext = ".azavg.txt"
data_q_ext = ".q.csv"
data_r_ext = ".r.csv"
image_ext = ".img.tiff"

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


def make_analyse_folder(dc_filepath):
    if os.path.isdir(dc_filepath):
        analysis_folder = os.path.join(dc_filepath, analysis_folder_name)
    else:
        current_folder, current_file_full_name = os.path.split(dc_filepath)
        analysis_folder = os.path.join(current_folder, analysis_folder_name)
    if not os.path.isdir(analysis_folder):
        try:
            os.makedirs(analysis_folder)
            return analysis_folder
        except:
            print('Failed to make directory:', analysis_folder)
            return False
    return analysis_folder


def save_current_azimuthal(data: np.ndarray, current_file_path, azavg: bool, i_slice=None):
    assert type(data) is np.ndarray
    current_folder_path, file_name = os.path.split(current_file_path)
    file_short_name, file_ext = os.path.splitext(file_name)

    analysis_folder_path = make_analyse_folder(current_file_path)
    if not analysis_folder_path:
        return

    if azavg:
        path_save = os.path.join(analysis_folder_path, file_short_name + " azav")
    else:
        path_save = os.path.join(analysis_folder_path, file_short_name + " azvar")
    if i_slice:
        path_save = path_save + " center" + str(i_slice[0]) + "to" + str(i_slice[1]) + "_" + str(i_slice[2])

    # add extension
    path_save = path_save + ".txt"

    np.savetxt(path_save, data)
    print("save to", path_save)


def load_preset_default(dc_file_path):
    current_folder_path, file_name = os.path.split(dc_file_path)
    file_short_name, file_ext = os.path.splitext(file_name)

    analysis_folder_path = os.path.join(current_folder_path, analysis_folder_name)
    preset_path = os.path.join(analysis_folder_path, file_short_name + preset_ext)
    if os.path.isfile(preset_path):
        return json.load(preset_path)
    else:
        return False


def save_preset_default(datacube, imgPanel=None):
    if datacube.preset_file_path is not None:
        # When you load preset files
        current_folder_path, file_name = os.path.split(datacube.preset_file_path)
        file_short_name = str(file_name)[:str(file_name).find(preset_ext)]
        analysis_folder_path = current_folder_path
    elif datacube.load_file_path is None:
        # When there is no source ( when load from other program )
        fp, _ = QFileDialog.getSaveFileName(filter="preset Files (*.preset.json)")
        current_folder_path = file_name = os.path.split(fp)
        file_short_name = str(file_name)[:str(file_name).find(preset_ext)]
        analysis_folder_path = current_folder_path
    else:
        # When you load only azavg or img files
        current_folder_path, file_name = os.path.split(datacube.load_file_path)
        file_short_name, file_ext = os.path.splitext(file_name)
        analysis_folder_path = make_analyse_folder(datacube.load_file_path)

    preset_path = os.path.join(analysis_folder_path, file_short_name + preset_ext)
    azavg_path = os.path.join(analysis_folder_path, file_short_name + azavg_ext)
    data_q_path = os.path.join(analysis_folder_path, file_short_name + data_q_ext)
    data_r_path = os.path.join(analysis_folder_path, file_short_name + data_r_ext)
    img_path = os.path.join(analysis_folder_path, file_short_name + image_ext)

    # save azavg
    if datacube.azavg is not None:
        np.savetxt(azavg_path, datacube.azavg)
    # save q data
    if datacube.q is not None:
        df = pd.DataFrame(
            {'q': datacube.q, 'Iq': datacube.Iq, 'Autofit': datacube.Autofit, 'phiq': datacube.phiq,
             'phiq_damp': datacube.phiq_damp})
        df.to_csv(data_q_path, index=None)
    # save r data
    if datacube.r is not None:
        df = pd.DataFrame({'r': datacube.r, 'Gr': datacube.Gr})
        df.to_csv(data_r_path, index=None)
    # save img data
    if imgPanel is not None and datacube.img is not None:
        imgPanel.imageView.export(img_path)

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
    json.dump(presets, open(preset_path, 'w'), indent=2)
    return True


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

    dc.load_file_path = fp
    dc.preset_file_path = fp
    if os.path.isfile(azavg_path):
        df_azavg = np.loadtxt(azavg_path)
        dc.azavg = df_azavg
    if os.path.isfile(data_r_path):
        df_r = pd.read_csv(data_r_path)
        dc.r = df_r['r'].to_numpy()
        dc.Gr = df_r['Gr'].to_numpy()
    if os.path.isfile(data_q_path):
        df_q = pd.read_csv(data_q_path)
        dc.q = df_q['q'].to_numpy()
        dc.Iq = df_q['Iq'].to_numpy()
        dc.phiq = df_q['phiq'].to_numpy()
        dc.phiq_damp = df_q['phiq_damp'].to_numpy()
        dc.Autofit = df_q['Autofit'].to_numpy()

    # convert relative path to absolute path
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
    # save_current_azimuthal(np.array([1,2,3]),pth,True)
    from PyQt5.QtWidgets import QMainWindow
    from PyQt5 import QtWidgets

    qtapp = QtWidgets.QApplication([])
    # load_azavg_manual(r"Y:\experiment\TEM diffraction\210520\Analysis pdf_tools\Camera 230 mm Ceta 20210520 1306_Au azav center110to120_1.txt")
    load_azavg_manual()
    qtapp.exec_()
    pass
