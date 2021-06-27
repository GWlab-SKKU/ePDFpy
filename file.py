import mrcfile
import os
import numpy as np

analysis_folder_name = "Analysis pdf_tools"

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


def save_current_azimuthal(data:np.ndarray,current_file_path,azavg,i_slice=None):
    assert type(data) is np.ndarray
    current_folder, current_file_full_name = os.path.split(current_file_path)
    current_file_name,current_ext = os.path.splitext(current_file_full_name)
    analysis_folder = os.path.join(current_folder,analysis_folder_name)
    if not os.path.isdir(analysis_folder):
        try :
            os.makedirs(analysis_folder)
        except:
            print('Failed to make directory:',analysis_folder)
            return
    if azavg:
        path_save = os.path.join(analysis_folder, current_file_name + "_azav")
    else:
        path_save = os.path.join(analysis_folder, current_file_name + "_azvar")
    if i_slice:
        path_save = path_save+str(i_slice[0])+"to"+str(i_slice[1])+"_"+str(i_slice[2])

    # add extension
    path_save = path_save+".txt"

    np.savetxt(path_save,data)
    print("save to",path_save)


if __name__ == '__main__':
    # file_list = get_file_list_from_path('/mnt/experiment/TEM diffraction/210312','.mrc')
    # print(os.path.split(file_list[0]))
    # print(get_file_list_from_path('/mnt/experiment/TEM diffraction/210312','.mrc'))
    # pth="/mnt/experiment/TEM diffraction/210215/sample47_TiGe44_bot_AD/Camera 230 mm Ceta 20210215 1438_2s_1f_area01.mrc"
    # print(os.path.split(pth))
    # save_current_azimuthal(np.array([1,2,3]),pth,True)
    pass