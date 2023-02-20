from PyQt5.QtWidgets import *
from ui.roi_selector import roi_selector
import numpy as np
import definitions
import hyperspy.api as hs
import mrcfile


# def sample_image_load():
#     # 4dSTEM
#     d4stem = False
#     if d4stem is True:
#         fp = r"/home/carrotrabbit/Documents/Berkeley/1_20x100_ss=5nm_C2=40um_alpha=p73urad_spot11_500ms_CL=245_bin=4_300kV.dm3"
#         # fp = r"C:\Users\vlftj\Documents\FEM\1_20x100_ss=5nm_C2=40um_alpha=p73urad_spot11_500ms_CL=245_bin=4_300kV.dm3"
#         file = hs.load(fp)
#         dc = file.data
#         std_img = np.std(dc, axis=0)
#         easy_img = np.log(np.abs(std_img) + 1)
#         return easy_img
#     else:
#         fp = r"/mnt/experiment/TEM diffraction/220104/Sample85_ZrTa_800C_postB/Camera 230 mm Ceta 20220104 1627_40s_20f_area21.mrc"
#         with mrcfile.open(fp) as mrc:
#             data = mrc.data
#         easy_img = np.log(np.abs(data) + 1)
#         return easy_img

def sample_image_load():
    fp = r"C:\\Users\\vlftj\\Documents\\sample43_TiTa_AD\\Camera 230 mm Ceta 20201126 1433_40s_20f_area01.mrc"
    with mrcfile.open(fp) as mrc:
        data = mrc.data
    return data

def fem_sample_load():
    # fp = r'/mnt/experiment/TEM diffraction/2022Berkeley02/FEM data/20220204/sample6_postE/220204_5_aSi_AD_postE_30x20_ss=5nm_C2=40um_alpha=0p63urad_spot11_500ms_CL=245_bin=4_300kV.dm4'
    fp = r'V:\experiment\TEM diffraction\2022Berkeley02\FEM data\20220204\sample6_postE\220204_5_aSi_AD_postE_30x20_ss=5nm_C2=40um_alpha=0p63urad_spot11_500ms_CL=245_bin=4_300kV.dm4'
    file = hs.load(fp)
    dc = file.data
    std_img = np.std(dc, axis=0)
    return std_img

if __name__ == '__main__':
    app = QApplication([])
    qw = QWidget()
    qw.layout = QHBoxLayout()
    qw.setLayout(qw.layout)
    img = sample_image_load()
    print("load finished")
    import mask_module
    module = mask_module.MaskModule()
    module.update_img(img)
    qw.layout.addWidget(module.dropdown)
    qw.show()
    # roi_selector.RoiCreater(img).show()


    app.exec_()


