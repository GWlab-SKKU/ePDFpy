import pyqtgraph as pg
import PyQt5
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
import sys
import hyperspy.api as hs
import numpy as np
import cv2
import os

# w2 = w.addLayout(row=0, col=1)
# label2 = w2.addLabel(text, row=0, col=0)
# v2a = w2.addViewBox(row=1, col=0, lockAspect=True)
# r2a = pg.PolyLineROI([[0,0], [10,10], [10,30], [30,10]], closed=True)
# v2a.addItem(r2a)

# def init_ui():
#     # widget = QtWidgets.QWidget()
#     # layout = QtWidgets.QHBoxLayout()
#     # layout.addWidget(QtWidgets.QPushButton(""))
#
#     # pg.ImageView()
#     # layout.addWidget(gwidget)
#     # w1 = gwidget.addLayout(row=0, col=0)
#     #
#     # v2a = w1.addViewBox(row=1, col=0, lockAspect=True)
#     # r2a = pg.PolyLineROI([[0,0], [10,10], [10,30], [30,10]], closed=True)
#     # v2a.addItem(r2a)
#     window = QtWidgets.QWidget()
#     layout = QtWidgets.QVBoxLayout()
#     layout.addWidget(QtWidgets.QPushButton('Top'))
#     layout.addWidget(QtWidgets.QPushButton('Bottom'))
#     window.setLayout(layout)
#     window.show()
#
#     # widget.setLayout(layout)
#     # widget.show()
#
#
class maskWidget(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        layout = QtWidgets.QVBoxLayout()
        self.imageView = pg.ImageView()

        layout_btn = QtWidgets.QHBoxLayout()
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        layout_btn.addWidget(self.btn_ok)
        layout_btn.addWidget(self.btn_cancel)

        layout.addWidget(self.imageView)
        layout.addLayout(layout_btn)
        self.setLayout(layout)
        self.setMinimumSize(500,500)

        self.binding()

    def image_load(self, img):
        self.imageView.setImage(img)
        pass

    def draw_roi(self):
        self.poly_line_roi = pg.PolyLineROI([[0,0], [10,10], [10,30], [30,10]], closed=True)
        self.imageView.addItem(self.poly_line_roi)
        pass

    def binding(self):
        self.btn_ok.clicked.connect(self.btn_ok_clicked)
        self.btn_cancel.clicked.connect(self.btn_cancel_clicked)

    def btn_ok_clicked(self):
        handles = [handle.pos() for handle in self.poly_line_roi.getHandles()]
        handles = np.array(handles)
        img = np.zeros(self.imageView.image.shape)
        cv2.fillPoly(img, pts=[handles.astype(np.int)], color=(255, 255, 255))

        fp, _ = QFileDialog.getSaveFileName()
        if fp is "":
            return
        if os.path.splitext(fp)[1] is None or os.path.splitext(fp)[1] != ".csv":
            fp = fp + ".csv"
        np.savetxt(fp, img, delimiter=',', fmt='%s')
        print("save to {}".format(fp))
        return

    def btn_cancel_clicked(self):
        return

def sample_image_load():
    # fp = r"/mnt/experiment/TEM diffraction/2021Berkeley/FEM data/20211005/1_20x100_ss=5nm_C2=40um_alpha=p73urad_spot11_500ms_CL=245_bin=4_300kV.dm3"
    fp = r"C:\Users\vlftj\Documents\FEM\1_20x100_ss=5nm_C2=40um_alpha=p73urad_spot11_500ms_CL=245_bin=4_300kV.dm3"
    file = hs.load(fp)
    dc = file.data
    std_img = np.std(dc, axis=0)
    easy_img = np.log(np.abs(std_img) + 1)
    return easy_img

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = maskWidget()
    window.show()
    window.image_load(sample_image_load())
    window.draw_roi()
    app.exec()




