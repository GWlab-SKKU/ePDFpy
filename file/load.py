import os, sys
import hyperspy.api as hs
import mrcfile
import numpy as np
from PIL import Image
import logging
import h5py

logger = logging.getLogger("Loader")

def load_diffraction_image(fp):
    if not os.path.isfile(fp):
        logger.error(f"{fp} is not a file")
        return
    ext = os.path.splitext(fp)[1]
    if ext in ['.jpg', '.jpeg', '.tiff', '.png']:
        raw_img = load_PIL_img(fp)
    elif ext in ['.mrc']:
        raw_img = load_mrc_img(fp)
    elif ext in ['.dm3', '.dm4']:
        raw_img = load_dm_img(fp)
    elif ext in ['.txt', '.csv']:
        raw_img = load_txt_img(fp)
    elif ext in ['.h5']:
        raw_img = load_h5_image(fp)
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
    if ext in ['.dm3','.dm4']:
        raw_img = load_dm_img(fp)
    elif ext in ['.h5']:
        raw_img = load_h5_image(fp)
    else:
        logger.error(f"Error, Non support data type: {ext}")
        raw_img = None
    logger.info(f"Load '{fp}', shape {str(raw_img.shape)}")
    return raw_img


def load_h5_image(fp):
    f = h5py.File(fp, 'r')
    data = np.array(f['4DSTEM_experiment']['data']['datacubes']['datacube_0']['data'])
    # todo:
    return data


def load_dm_img(fp):
    return hs.load(fp).data


def load_txt_img(fp):
    ext = os.path.splitext(fp)[1]
    if ext == '.csv':
        raw_img = np.loadtxt(fp)
    if ext == '.txt':
        raw_img = np.loadtxt(fp)
    return raw_img


def load_mrc_img(fp):
    print("Loading file:",fp)
    with mrcfile.open(fp) as mrc:
        raw_img = mrc.data
    return raw_img


def load_PIL_img(path):
    img = Image.open(path).convert("L")
    raw_img = np.array(img)
    return raw_img