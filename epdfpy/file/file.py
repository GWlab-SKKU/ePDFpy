import mrcfile
import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog
import json
import copy
import pandas as pd
from pathlib import Path
from epdfpy import definitions
import re
import cv2
from PIL import Image
import hyperspy.api as hs
from epdfpy.datacube.cube import PDFCube

ePDFpy_analysis_folder_name = "Analysis ePDFpy"
preset_ext = ".preset.json"
azavg_ext = ".azavg.txt"
normstd_ext = ".normstd.txt"
data_q_ext = ".q.csv"
data_r_ext = ".r.csv"
image_ext = ".img.png"
rdf_screen_ext = ".rdf.png"


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

def save_preset(datacubes, main_window, fpth, stack=True, saveas=True):
    imgPanel = main_window.profile_extraction.img_panel

    for datacube in datacubes:
        load_folder_path, load_file_name = os.path.split(datacube.load_file_path)
        file_short_name, load_file_ext = os.path.splitext(load_file_name)
        if '.azavg' in file_short_name:
            file_short_name = file_short_name.replace('.azavg','')
        if '.preset' in file_short_name:
            file_short_name = file_short_name.replace('.preset','')

        if (stack) and (saveas):
            sample_folder = os.path.join(fpth, f"({datacube.data_quality}){file_short_name}")
            os.makedirs(sample_folder, exist_ok=True)
        if (not stack) and (saveas):
            sample_folder = fpth
            os.makedirs(sample_folder, exist_ok=True)
        if (stack) and (not saveas):
            current_sample_folder = os.path.split(datacube.preset_file_path)[0]
            upper_folder, sample_folder_short = os.path.split(current_sample_folder)
            new_sample_folder_short = re.sub(r'\(L.\)',"({})".format(datacube.data_quality), sample_folder_short)
            new_sample_folder_short = re.sub(r'\(None\)',"({})".format(datacube.data_quality), new_sample_folder_short)
            sample_folder = os.path.join(upper_folder,new_sample_folder_short)
            if sample_folder_short != new_sample_folder_short:
                os.rename(current_sample_folder, sample_folder)
        if (not stack) and (not saveas):
            current_sample_folder = os.path.split(datacube.preset_file_path)[0]
            upper_folder, sample_folder_short = os.path.split(current_sample_folder)
            new_sample_folder_short = re.sub(r'\(L.\)',"({})".format(datacube.data_quality), sample_folder_short)
            new_sample_folder_short = re.sub(r'\(None\)',"({})".format(datacube.data_quality), new_sample_folder_short)
            sample_folder = os.path.join(upper_folder,new_sample_folder_short)
            if sample_folder_short != new_sample_folder_short:
                os.rename(current_sample_folder, sample_folder)

        preset_path = os.path.join(sample_folder, file_short_name + preset_ext)
        data_q_path = os.path.join(sample_folder, file_short_name + data_q_ext)
        data_r_path = os.path.join(sample_folder, file_short_name + data_r_ext)
        img_path = os.path.join(sample_folder, file_short_name + image_ext)
        rdf_screen_path = os.path.join(sample_folder, file_short_name + rdf_screen_ext)
        datacube.preset_file_path = preset_path

        # full q data
        if datacube.full_q is not None:
            # q, Iq, Autofit, phiq, phiq_damp
            df_q = pd.DataFrame({'q':datacube.full_q})
            df_q['Iq'] = datacube.azavg

            lst = ['Autofit', 'phiq', 'phiq_damp']
            for l in lst:
                df_q[l] = np.nan
                df_q[l][datacube.pixel_start_n:datacube.pixel_end_n+1] = getattr(datacube, l)
            df_q.to_csv(data_q_path, index=None)

        # save r data
        if datacube.r is not None:
            lst = ['r', 'Gr']
            df_r = pd.DataFrame({name: getattr(datacube, name) for name in lst if getattr(datacube, name) is not None})
            df_r.to_csv(data_r_path, index=None)
        # save img data
        # if imgPanel is not None and datacube.display_img is not None:
        #     imgPanel.imageView.export(img_path)
        # save screenshot
        if datacube.Gr is not None:
            main_window.PDF_analyser.grab().save(rdf_screen_path)

        ################ save preset #################
        presets = vars(copy.copy(datacube))

        # # convert to relative path
        # try:
        #     # deprecated: mrc_file_path
        #     if presets['img_file_path'] is not None:
        #         presets['img_file_path'] = os.path.relpath(datacube.img_file_path, os.path.split(preset_path)[0])
        #         presets['img_file_path'] = presets['img_file_path'].replace('\\','/')  # compatibility between windows and linux
        # except:
        #     if presets['mrc_file_path'] is not None:
        #         presets['mrc_file_path'] = os.path.relpath(datacube.img_file_path, os.path.split(preset_path)[0])
        #         presets['mrc_file_path'] = presets['mrc_file_path'].replace('\\','/')  # compatibility between windows and linux

        # remove data that not support to save as json
        for key, value in dict(presets).items():
            if type(value) not in [int, str, float, list, np.float64, np.float32, np.int64, np.int32, np.uint]:
                del presets[key]

        # Don't save in preset file
        remove_list = ['load_file_path', 'preset_file_path']
        for remove_item in remove_list:
            if remove_item in dict(presets).keys():
                del presets[remove_item]

        # type exception handling
        presets = type_changer(presets)

        # # deprecated
        # if presets['center'][0] is not None:
        #     presets['center'] = [int(presets['center'][0]), int(presets['center'][1])]

        print("save data:", presets)
        print(preset_path)
        json.dump(presets, open(preset_path, 'w'), indent=2)
        return True


def type_changer(items):
    if type(items) == dict:
        for k,v in items.items():
            if type(v) in [np.int64, np.int32, np.uint]:
                items[k] = int(v)
            if type(v) in [np.float64, np.float32]:
                items[k] = float(v)
            if type(v) == dict or type(v) == list:
                items[k] = type_changer(v)
    elif type(items) == list:
        tmp_list = []
        for v in items:
            if type(v) in [np.int64, np.int32, np.uint]:
                v = int(v)
            if type(v) in [np.float64, np.float32]:
                v = float(v)
            if type(v) == dict or type(v) == list:
                v = type_changer(v)
            tmp_list.append(v)
        items = tmp_list
    return items


def load_preset(fp:str=None, dc=None):
    if fp is None:
        fp, _ = QFileDialog.getOpenFileName(filter="preset Files (*.preset.json)")
    if fp == '':
        return

    if dc is None:
        dc = PDFCube()
    try:
        content = json.load(open(fp))
    except:
        print(f"Error while loading preset file: {fp}")

    data_r_path = os.path.join(os.path.split(fp)[0], fp[:fp.rfind(preset_ext)] + data_r_ext)
    data_q_path = os.path.join(os.path.split(fp)[0], fp[:fp.rfind(preset_ext)] + data_q_ext)
    # normstd_path = os.path.join(os.path.split(fp)[0], fp[:fp.rfind(preset_ext)] + normstd_ext)

    dc.load_file_path = fp
    dc.preset_file_path = fp

    # put content in DataCube
    for key, value in content.items():
        if key in vars(dc).keys():
            setattr(dc, key, value)

    # previous q
    if os.path.isfile(data_q_path):
        df_q = pd.read_csv(data_q_path)
        for column in df_q.columns:
            setattr(dc, column, df_q[column].to_numpy())
    azavg_path = os.path.join(os.path.split(fp)[0], fp[:fp.rfind(preset_ext)] + azavg_ext)
    if os.path.isfile(azavg_path):
        df_azavg = np.loadtxt(azavg_path)
        dc.azavg = df_azavg

    # current version of q
    if (dc.pixel_end_n - dc.pixel_start_n+1) != len(df_q):
        df_q = pd.read_csv(data_q_path)
        dc.full_q = np.array(df_q['q'])
        dc.azavg = np.array(df_q['Iq'])
        dc.Iq = np.array(dc.azavg[dc.pixel_start_n:dc.pixel_end_n+1])
        dc.q = np.array(dc.full_q[dc.pixel_start_n:dc.pixel_end_n+1])
        dc.phiq = np.array(df_q['phiq'][dc.pixel_start_n:dc.pixel_end_n+1])
        dc.phiq_damp = np.array(df_q['phiq_damp'][dc.pixel_start_n:dc.pixel_end_n+1])
        dc.Autofit = np.array(df_q['Autofit'][dc.pixel_start_n:dc.pixel_end_n+1])

    if os.path.isfile(data_r_path):
        df_r = pd.read_csv(data_r_path)
        for column in df_r.columns:
            setattr(dc, column, df_r[column].to_numpy())

    # # deprecated
    # if os.path.isfile(normstd_path):
    #     df_normstd = np.loadtxt(normstd_path)
    #     dc.azvar = df_normstd

    # deprecated: img load
        #     deprecated: mrc_file_path
        #     if 'mrc_file_path' in content.keys():
        #         content['mrc_file_path'] = os.path.abspath(os.path.join(fp, "..", content['mrc_file_path']))
        #         content['mrc_file_path'] = os.path.abspath(os.path.join(fp, "..", content['mrc_file_path']))
        #     # convert relative path to absolute path
        #     if 'img_file_path' in content.keys():
        #         content['img_file_path'] = os.path.abspath(os.path.join(fp, "..", content['img_file_path']))
        #         content['img_file_path'] = os.path.abspath(os.path.join(fp, "..", content['img_file_path']))

    return dc


def save_pdf_setting_manual(dc_file_path):
    fp, _ = QFileDialog.getSaveFileName()
    json.dump(fp, open(fp, 'w'), indent=2)
    return True


def load_element_preset():
    if not os.path.isfile(definitions.ELEMENT_PRESETS_PATH):
        element_preset = dict()
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
    if fp == '':
        return
    if 'csv' in ext:
        if fp[-4:] == ".csv":
            np.savetxt(fp, azavg, delimiter=',')
        else:
            np.savetxt(fp+".csv", azavg, delimiter=',')
    elif 'txt' in ext:
        if fp[-4:] == ".txt":
            np.savetxt(fp, azavg)
        else:
            np.savetxt(fp + ".txt", azavg)

def save_azavg_stack(dcs):
    dirpth = QFileDialog.getExistingDirectory(None, '')
    for dc in dcs:
        if dc.azavg is None:
            continue
        filename = os.path.split(dc.load_file_path)[1]
        short_name, ext = os.path.splitext(filename)
        fp = os.path.join(dirpth, short_name+".azavg.csv")
        np.savetxt(fp, dc.azavg, delimiter = ',')

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
