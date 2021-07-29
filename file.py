import mrcfile
import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog
import json
from datacube import DataCube
import copy
import pandas as pd
from pathlib import Path

analysis_folder_name = "Analysis ePDFpy"
preset_ext = ".preset.json"
azavg_ext = ".azavg.csv"
data_q_ext = ".q.csv"
data_r_ext = ".r.csv"


def load_mrc_img(fp):
    with mrcfile.open(fp) as mrc:
        raw_img = mrc.data
    easy_img = np.log(np.abs(raw_img) + 1)
    easy_img = easy_img / easy_img.max() * 255
    easy_img = easy_img.astype('uint8')
    return raw_img, easy_img


def get_file_list_from_path2(fp, extension=None):
    if type(extension) is str:
        extension = [extension]
    if not os.path.isdir(fp):
        return
    file_list = []
    for (path, dir, files) in os.walk(fp):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if extension is not None:
                if ext in extension:
                    file_list.append(os.path.join(path, filename))
    return file_list


def get_file_list_from_path(fp, extension=None):
    files = Path(fp).rglob("*" + extension)
    file_list = []
    for _file in files:
        file_list.append(str(_file.absolute()))
    return file_list


def make_analyse_folder(dc_filepath):
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


def load_preset_manual():
    fp, _ = QFileDialog.getOpenFileName()
    return json.load(fp)


def save_preset_default(dc_file_path, datacube):
    if dc_file_path is None and datacube.azavg_file_path is not None:
        # when you load only azavg
        current_folder_path, file_name = os.path.split(datacube.azavg_file_path)
        file_short_name, file_ext = os.path.splitext(file_name)
        analysis_folder_path = make_analyse_folder(dc_file_path)

    elif dc_file_path is None and datacube.azavg_file_path is None:
        # if come from averaging multiple
        fp, _ = QFileDialog.getSaveFileName(filter="preset Files (*.preset.json)")
        current_folder_path, file_name = os.path.split(fp)
        file_short_name, file_ext = os.path.splitext(file_name)
        analysis_folder_path = current_folder_path

    else:
        current_folder_path, file_name = os.path.split(dc_file_path)
        file_short_name, file_ext = os.path.splitext(file_name)
        analysis_folder_path = make_analyse_folder(dc_file_path)


    preset_path = os.path.join(analysis_folder_path, file_short_name + preset_ext)
    azavg_path = os.path.join(analysis_folder_path, file_short_name + azavg_ext)
    data_q_path = os.path.join(analysis_folder_path, file_short_name + data_q_ext)
    data_r_path = os.path.join(analysis_folder_path, file_short_name + data_r_ext)

    to_upload = vars(copy.copy(datacube))

    # save azavg
    if to_upload['azavg'] is not None:
        np.savetxt(azavg_path, to_upload['azavg'])
    # save q data
    if to_upload['q'] is not None:
        df = pd.DataFrame(
            {'q': to_upload['q'], 'Iq': to_upload['Iq'], 'Autofit': to_upload['Autofit'], 'phiq': to_upload['phiq'],
             'phiq_damp': to_upload['phiq_damp']})
        df.to_csv(data_q_path, index=None)
    # save r data
    if to_upload['r'] is not None:
        df = pd.DataFrame({'r': to_upload['r'], 'Gr': to_upload['Gr']})
        df.to_csv(data_r_path, index=None)

    # convert to relative path
    if to_upload['mrc_file_path'] is not None:
        to_upload['mrc_file_path'] = os.path.relpath(dc_file_path, os.path.split(preset_path)[0])

    print(to_upload['fit_at_q'])
    print("type:{}".format(type(to_upload['fit_at_q'])))

    to_upload2 = {}

    # remove data that not support to save as json
    for key, value in to_upload.items():
        if type(value) in [int, str, float, list, np.float64, np.int64]:
            to_upload2.update({key: value})

    # int64 exception handling
    for key, value in to_upload.items():
        if type(value) == np.int64:
            to_upload2[key] = int(value)
        if type(value) == np.float64:
            to_upload2[key] = float(value)

    if to_upload2['center'][0] is not None:
        to_upload2['center'] = [int(to_upload2['center'][0]), int(to_upload2['center'][1])]

    print("save data:", to_upload2)

    json.dump(to_upload2, open(preset_path, 'w'), indent=2)
    return True


def load_preset(fp: str = None):
    if fp is None:
        fp, _ = QFileDialog.getOpenFileName(filter="preset Files (*.preset.json)")
    if fp == '':
        return

    dc = DataCube()
    content = json.load(open(fp))

    azavg_path = os.path.join(os.path.split(fp)[0], fp[:fp.rfind(preset_ext)] + azavg_ext)
    data_r_path = os.path.join(os.path.split(fp)[0], fp[:fp.rfind(preset_ext)] + data_r_ext)
    data_q_path = os.path.join(os.path.split(fp)[0], fp[:fp.rfind(preset_ext)] + data_q_ext)

    if os.path.isfile(azavg_path):
        df_azavg = np.loadtxt(azavg_path)
        dc.azavg = df_azavg
    if os.path.isfile(data_r_path):
        df_r = pd.read_csv(data_r_path)
        dc.r = df_r['r']
        dc.Gr = df_r['Gr']
    if os.path.isfile(data_q_path):
        df_q = pd.read_csv(data_q_path)
        dc.q = df_q['q']
        dc.Iq = df_q['Iq']
        dc.phiq = df_q['phiq']
        dc.phiq_damp = df_q['phiq_damp']
        dc.Autofit = df_q['Autofit']

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


# def load_azavg_from_preset(preset_path:str):
#     preset_path.rfind(preset_ext)


def load_azavg_manual():
    fp, _ = QFileDialog.getOpenFileName()
    if fp is '':
        return
    azavg = load_azavg(fp)
    dc = DataCube()
    dc.azavg = azavg
    dc.azavg_file_path = fp
    return load_azavg(fp)


def load_azavg(fp) -> np.ndarray:
    current_folder_path, file_name = os.path.split(fp)
    file_short_name, file_ext = os.path.splitext(file_name)
    if file_ext == ".csv":
        return np.loadtxt(fp, delimiter=",")
    if file_ext == ".txt":
        return np.loadtxt(fp)


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
