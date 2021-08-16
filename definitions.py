import os, sys, pathlib
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_JSON_PATH = os.path.join(ROOT_DIR, 'settings/default.json')
ELEMENT_PRESETS_PATH = os.path.join(ROOT_DIR, 'settings/element_presets.json')

KIRKLAND_PATH = os.path.join(ROOT_DIR, 'assets/Parameter_files/Kirkland_2010.txt')
LOBATO_PATH = os.path.join(ROOT_DIR, 'assets/Parameter_files/Lobato_2014.txt')

MASK_PATH = os.path.join(ROOT_DIR, 'assets/mask_data2.txt')
COLORCUBE = os.path.join(ROOT_DIR, 'assets/colorcube256.csv')
ATOMIC_SYMBOL_PATH = os.path.join(ROOT_DIR, "assets/Parameter_files/atomicNumber-symbol.csv")

SAMPLE_IMG_PATH = os.path.join(ROOT_DIR, "assets/Camera 230 mm Ceta 20210312 1333_50s_20f_area01.mrc")