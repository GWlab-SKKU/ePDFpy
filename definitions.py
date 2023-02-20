import os, sys, pathlib
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PROGRAM_NAME = "ePDFpy"

DEFAULT_JSON_PATH = os.path.join(ROOT_DIR, 'settings/default.json')
ELEMENT_PRESETS_PATH = os.path.join(ROOT_DIR, 'settings/element_presets.json')
DATA_QUALITY_PATH = os.path.join(ROOT_DIR, 'settings/data_quality.csv')

KIRKLAND_PATH = os.path.join(ROOT_DIR, 'assets/Parameter_files/Kirkland_2010.txt')
LOBATO_PATH = os.path.join(ROOT_DIR, 'assets/Parameter_files/Lobato_2014.txt')

MASK_PATH = os.path.join(ROOT_DIR, 'assets/mask_data.json')
MASK_PATH_DEFAULT = os.path.join(ROOT_DIR, 'assets/mask_data.txt')
MASK_FOLDER_PATH = os.path.join(ROOT_DIR, 'assets/mask')

COLORCUBE = os.path.join(ROOT_DIR, 'assets/colorcube256.csv')
ATOMIC_SYMBOL_PATH = os.path.join(ROOT_DIR, "assets/Parameter_files/atomicNumber-symbol.csv")

SAMPLE_IMG_PATH = os.path.join(ROOT_DIR, "assets/Camera 230 mm Ceta 20210312 1333_50s_20f_area01.mrc")

STYLE_PATH = os.path.join(ROOT_DIR, 'assets/css/STYLE.qss')
THEME_PATH = os.path.join(ROOT_DIR, 'assets/css/Combinear.qss')

THEME_PATH2 = os.path.join(ROOT_DIR, 'assets/css/material_dark.css')

