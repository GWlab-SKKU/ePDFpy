import mrcfile
import os
import numpy as np


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

if __name__ == '__main__':
    file_list = get_file_list_from_path('/mnt/experiment/TEM diffraction/210312','.mrc')
    print(os.path.split(file_list[0]))
    # print(get_file_list_from_path('/mnt/experiment/TEM diffraction/210312','.mrc'))
