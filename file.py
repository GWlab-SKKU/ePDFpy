import mrcfile
import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog
import json

analysis_folder_name = "Analysis ePDFpy"

def load_mrc_img(fp):
    with mrcfile.open(fp) as mrc:
        raw_img = mrc.data
    easy_img = np.log(np.abs(raw_img)+1)
    easy_img = easy_img / easy_img.max() * 255
    easy_img = easy_img.astype('uint8')
    return raw_img, easy_img

def get_file_list_from_path(fp, extension=None):
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

def make_analyse_folder(dc_filepath):
    current_folder, current_file_full_name = os.path.split(dc_filepath)
    analysis_folder = os.path.join(current_folder,analysis_folder_name)
    if not os.path.isdir(analysis_folder):
        try :
            os.makedirs(analysis_folder)
            return analysis_folder
        except:
            print('Failed to make directory:',analysis_folder)
            return False
    return analysis_folder

def save_current_azimuthal(data:np.ndarray,current_file_path,azavg:bool,i_slice=None):
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
        path_save = path_save+" center"+str(i_slice[0])+"to"+str(i_slice[1])+"_"+str(i_slice[2])

    # add extension
    path_save = path_save+".txt"

    np.savetxt(path_save,data)
    print("save to",path_save)


def load_preset_default(dc_file_path):
    current_folder_path, file_name = os.path.split(dc_file_path)
    file_short_name, file_ext = os.path.splitext(file_name)

    analysis_folder_path = os.path.join(current_folder_path,analysis_folder_name)
    preset_path = os.path.join(analysis_folder_path, file_short_name + " preset.json")
    if os.path.isfile(preset_path):
        return json.load(preset_path)
    else:
        return False

def load_preset_manual():
    fp, _ = QFileDialog.getOpenFileName()
    return json.load(fp)

def save_preset_default(dc_file_path, setting):
    current_folder_path, file_name = os.path.split(dc_file_path)
    file_short_name, file_ext = os.path.splitext(file_name)

    analysis_folder_path = make_analyse_folder(dc_file_path)
    preset_path = os.path.join(analysis_folder_path, file_short_name + " preset.json")
    if setting['mrc_file_path'] is not None:
        setting['mrc_file_path'] = os.path.relpath(dc_file_path, os.path.split(preset_path)[0])
    json.dump(setting, open(preset_path, 'w'), indent=2)
    return True

def load_preset():
    fp, _ = QFileDialog.getOpenFileName()
    return json.load(fp)

def save_pdf_setting_manual(dc_file_path):
    fp, _ = QFileDialog.getSaveFileName()
    json.dump(fp, open(fp, 'w'), indent=2)
    return True


def load_azavg_default(dc_file_path):
    pass

def load_azavg_manual():
    fp, _ = QFileDialog.getOpenFileName()
    current_folder_path, file_name = os.path.split(fp)
    file_short_name, file_ext = os.path.splitext(file_name)
    if file_ext == ".csv":
        return np.loadtxt(fp,delimiter=",")
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