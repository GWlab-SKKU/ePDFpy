import os, sys
import hyperspy.api as hs
import mrcfile
import numpy as np
from PIL import Image
import logging
import h5py

logger = logging.getLogger("Loader")


######### diffraction image extensions ##########
support_PIL_ext = ['.jpg', '.jpeg', '.tiff', '.png']
support_mrc_ext = ['.mrc']
support_dm_ext = ['.dm3', '.dm4']
support_txt_img = ['.txt', '.csv']
support_h5_ext = ['.h5']
support_diffraction_ext_list = []
support_diffraction_ext_list.extend([*support_PIL_ext,*support_mrc_ext,*support_dm_ext,*support_txt_img,*support_h5_ext])
############# stem image extensions #############
support_stem_ext_list = []
support_stem_ext_list.extend([*support_dm_ext,*support_h5_ext])


def load_diffraction_image(fp):
    if not os.path.isfile(fp):
        logger.error(f"{fp} is not a file")
        return
    ext = os.path.splitext(fp)[1]
    if ext in support_PIL_ext:
        raw_img = _load_PIL_img(fp)
    elif ext in support_mrc_ext:
        raw_img = _load_mrc_img(fp)
    elif ext in support_dm_ext:
        raw_img = _load_dm_img(fp)
    elif ext in support_txt_img:
        raw_img = _load_txt_img(fp)
    elif ext in support_h5_ext:
        raw_img = _load_h5_image(fp)
    else:
        logger.error(f"Error, Non support data type: {ext}")
        raw_img = None
    logger.info(f"Load '{fp}', shape {str(raw_img.shape)}")
    return raw_img


def load_stem_image(fp):
    if not os.path.isfile(fp):
        logger.error(f"{fp} is not a file")
        return
    ext = os.path.splitext(fp)[1]
    if ext in support_dm_ext:
        raw_img = _load_dm_img(fp)
    elif ext in support_h5_ext:
        raw_img = _load_h5_image(fp)
    else:
        logger.error(f"Error, Non support data type: {ext}")
        raw_img = None
    logger.info(f"Load '{fp}', shape {str(raw_img.shape)}")
    return raw_img


def _load_h5_image(fp):
    f = h5py.File(fp, 'r')
    data = np.array(f['4DSTEM_experiment']['data']['datacubes']['datacube_0']['data'])
    # todo:
    return data


def _load_dm_img(fp):
    return hs.load(fp).data


def _load_txt_img(fp):
    ext = os.path.splitext(fp)[1]
    if ext == '.csv':
        raw_img = np.loadtxt(fp)
    if ext == '.txt':
        raw_img = np.loadtxt(fp)
    return raw_img


def _load_mrc_img(fp):
    print("Loading file:",fp)
    with mrcfile.open(fp) as mrc:
        raw_img = mrc.data
    return raw_img


def _load_PIL_img(path):
    img = Image.open(path).convert("L")
    raw_img = np.array(img)
    return raw_img

