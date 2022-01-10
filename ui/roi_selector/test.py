from PyQt5.QtWidgets import *
from ui.roi_selector import roi_selector
import numpy as np
import definitions
import hyperspy.api as hs
import mrcfile


def sample_image_load():
    # 4dSTEM
    d4stem = False
    if d4stem is True:
        fp = r"/home/carrotrabbit/Documents/Berkeley/1_20x100_ss=5nm_C2=40um_alpha=p73urad_spot11_500ms_CL=245_bin=4_300kV.dm3"
        # fp = r"C:\Users\vlftj\Documents\FEM\1_20x100_ss=5nm_C2=40um_alpha=p73urad_spot11_500ms_CL=245_bin=4_300kV.dm3"
        file = hs.load(fp)
        dc = file.data
        std_img = np.std(dc, axis=0)
        easy_img = np.log(np.abs(std_img) + 1)
        return easy_img
    else:
        fp = r"/mnt/experiment/TEM diffraction/220104/Sample85_ZrTa_800C_postB/Camera 230 mm Ceta 20220104 1627_40s_20f_area21.mrc"
        with mrcfile.open(fp) as mrc:
            data = mrc.data
        easy_img = np.log(np.abs(data) + 1)
        return easy_img

if __name__ == '__main__':
    app = QApplication([])
    qw = QWidget()
    qw.layout = QHBoxLayout()
    qw.setLayout(qw.layout)
    img = sample_image_load()
    print("load finished")
    dropdown = roi_selector.MaskDropdown(image=img, mask_folder=definitions.MASK_FOLDER_PATH)
    qw.layout.addWidget(dropdown)

    qw.show()
    qw.update()

    app.exec_()


