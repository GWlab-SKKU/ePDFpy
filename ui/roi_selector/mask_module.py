import pyqtgraph as pg
import PyQt5
from PyQt5 import QtWidgets, Qt
from PyQt5.QtWidgets import QFileDialog
import sys
import hyperspy.api as hs
import numpy as np
import cv2
import os
import file
from pathlib import Path
from PyQt5.QtCore import QItemSelectionModel
import json
import definitions

class MaskModule():
    def __init__(self, img = None, imageView = None):
        self.img = img
        self.imageView = imageView
        self.mask_dict = {}

        self.roi_creator = RoiCreater(self)
        self.dropdown = DropDown(self)
        self.list_widget = ListWidget(self)

        self._mask_reload()

    def update_img(self, img):
        self.img = img
        self.dropdown.img = img
        self.list_widget.img = img
        self.roi_creator.img = img

    def get_current_mask(self):
        x_y = self.mask_dict[self.dropdown.currentText()]['data']
        return

    def _load_mask(self):
         self.image = json.load(definitions.MASK_PATH)

    def _mask_reload(self):
        self.mask_dict
        self.dropdown.mask_reload()
        # self.list_widget.mask_reload()


class RoiCreater(QtWidgets.QWidget):
    def __init__(self, module:MaskModule):
        QtWidgets.QWidget.__init__(self)
        self.module = module
        # self.setWindowFlags(self.windowFlags() | Qt.Qt.Window)

        layout = QtWidgets.QVBoxLayout()
        self.imageView = pg.ImageView()
        layout_bottom = QtWidgets.QHBoxLayout()
        self.lbl_name = QtWidgets.QLabel("Name:")
        self.lbl_name.setMaximumWidth(100)
        self.txt_name = QtWidgets.QTextEdit()
        self.txt_name.setMaximumHeight(30)
        self.txt_name.setMaximumWidth(200)
        self.btn_ok = QtWidgets.QPushButton("OK")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")

        grp_save = QtWidgets.QGroupBox("Save")
        layout_bottom.addWidget(self.lbl_name)
        layout_bottom.addWidget(self.txt_name)
        layout_bottom.addWidget(self.btn_ok)
        layout_bottom.addWidget(self.btn_cancel)
        grp_save.setLayout(layout_bottom)
        grp_save.setMaximumHeight(80)

        grp_view_mode = QtWidgets.QGroupBox("View mode")
        view_mode_layout = QtWidgets.QHBoxLayout()
        self.radio_raw = QtWidgets.QRadioButton("Raw")
        self.radio_root = QtWidgets.QRadioButton("Root")
        self.radio_log = QtWidgets.QRadioButton("Log")
        view_mode_layout.addWidget(self.radio_raw)
        view_mode_layout.addWidget(self.radio_root)
        view_mode_layout.addWidget(self.radio_log)
        grp_view_mode.setLayout(view_mode_layout)
        self.radio_raw.setChecked(True)

        layout.addWidget(self.imageView)
        layout.addWidget(grp_view_mode)
        layout.addWidget(grp_save)
        self.setLayout(layout)
        # self.setBaseSize(800,800)
        self.setMinimumSize(800,700)

        self.update_image()
        self.draw_roi()

        self.binding()

        self.setWindowTitle("Masking")

    def start(self, new:bool):
        print("start Hello")
        self.show()
        pnts = None
        if not new:
            current_text = self.module.list_widget.QList.currentItem().text()
            pnts = self.module.mask_dict[current_text]['data']
        self.draw_roi(pnts)

    def update_image(self, img=None):
        if img is None:
            img = self.module.img
        if not img:
            return
        if self.radio_raw.isChecked():
            self.imageView.setImage(img)
        if self.radio_root.isChecked():
            self.imageView.setImage(np.power(img,0.5))
        if self.radio_log.isChecked():
            self.imageView.setImage(np.log((np.abs(img)+1)))

    def draw_roi(self, pnts=None):
        if not self.module.img:
            return
        if not pnts:
            w = self.module.img.shape[0]
            h = self.module.img.shape[1]
            pnts = [[int(w / 2) - int(w / 10), int(h / 2) - int(h / 10)],
                    [int(w / 2) - int(w / 10), int(h / 2) + int(h / 10)],
                    [int(w / 2) + int(w / 10), int(h / 2) - int(h / 10)],
                    [int(w / 2) + int(w / 10), int(h / 2) + int(h / 10)]]
        self.poly_line_roi = pg.PolyLineROI(pnts, closed=True)
        self.imageView.addItem(self.poly_line_roi)

    def binding(self):
        self.btn_ok.clicked.connect(self.btn_ok_clicked)
        self.btn_cancel.clicked.connect(self.btn_cancel_clicked)
        self.radio_raw.toggled.connect(lambda x: self.update_image())
        self.radio_root.toggled.connect(lambda x: self.update_image())
        self.radio_log.toggled.connect(lambda x: self.update_image())

    def btn_ok_clicked(self):
        handles = [handle.pos() for handle in self.poly_line_roi.getHandles()]
        handles = np.array(handles)
        img = np.zeros(self.imageView.image.shape)
        cv2.fillPoly(img, pts=[handles.astype(np.int)], color=(255, 255, 255))

        if self.save_mask_folder is None:
            fp, _ = QFileDialog.getSaveFileName()
            if fp == "":
                return
        else:
            name = self.txt_name.toPlainText()
            fp = os.path.join(self.save_mask_folder,name)

        Path(fp).mkdir(parents=True, exist_ok=True)

        if os.path.splitext(fp)[1] is None or os.path.splitext(fp)[1] != ".csv":
            fp = fp + ".csv"
        np.savetxt(fp, img, delimiter=',', fmt='%s')
        print("save to {}".format(fp))
        if self.func_after_mask_selected is not None:
            self.func_after_mask_selected()
        self.close()
        return

    def btn_cancel_clicked(self):
        self.close()
        return


class DropDown(QtWidgets.QComboBox):
    def __init__(self, module:MaskModule):
        """
        Args:
            image: 2d numpy array or pyqtgraph.ImageViewer
            mask_save_folder_path: folder where mask array will be saved
            after_mask_selected: function that will be excu
        """
        QtWidgets.QComboBox.__init__(self)
        self.module = module
        self.img = module.img
        self.currentIndexChanged.connect(self.dropdown_event)

    def mask_reload(self):
        self.clear()
        self.addItem("None")
        self.addItems(self.module.mask_dict.keys())
        self.addItem("[Edit]")

    def dropdown_event(self, idx):
        if idx == len(self.module.mask_dict) + 1:
            # edit
            self.module.list_widget.show()

class ListWidget(QtWidgets.QWidget):
    def __init__(self, module:MaskModule):
        super().__init__()
        self.module = module
        self.QList = QtWidgets.QListWidget()
        self.items = list(self.module.mask_dict.keys())
        self.QList.addItems(self.items)
        # self.QList.itemDoubleClicked.connect()
        self.layout = QtWidgets.QGridLayout()
        self.setLayout(self.layout)

        self.btn_move_up = QtWidgets.QPushButton("△")
        self.btn_move_down = QtWidgets.QPushButton("▽")
        self.btn_del = QtWidgets.QPushButton("Del")
        self.btn_edit = QtWidgets.QPushButton("Edit")
        self.btn_new = QtWidgets.QPushButton("New")

        self.btn_move_up.clicked.connect(self.btn_move_up_clicked)
        self.btn_move_down.clicked.connect(self.btn_move_down_clicked)
        self.btn_del.clicked.connect(self.btn_del_clicked)
        self.btn_edit.clicked.connect(self.btn_edit_clicked)
        self.btn_new.clicked.connect(self.btn_new_clicked)

        self.layout.addWidget(self.QList, 0, 0, 5, 2)
        self.layout.addWidget(self.btn_move_up, 0, 2)
        self.layout.addWidget(self.btn_move_down, 1, 2)
        self.layout.addWidget(self.btn_del, 2, 2)
        self.layout.addWidget(self.btn_edit, 3, 2)
        self.layout.addWidget(self.btn_new, 4, 2)

    def btn_move_up_clicked(self):
        indexes = self.QList.selectedIndexes()
        if len(indexes) > 0:
            try:
                indexes.sort()
                first_row = indexes[0].row() - 1
                if first_row >= 0:
                    for idx in indexes:
                        if idx != None:
                            row = idx.row()
                            self.items.insert(row - 1, self.items[row])
                            self.items.pop(row + 1)
                            self.QList.clear()
                            self.QList.addItems(self.items)
            except Exception as e:
                print(e)
        else:
            print('Select at least one item from list!')

    def btn_move_down_clicked(self):
        max_row = len(self.items)
        indexes = self.QList.selectedIndexes()
        if len(indexes) > 0:
            try:
                indexes.sort()
                last_row = indexes[-1].row() + 1
                if last_row < max_row:
                    for idx in indexes:
                        if idx != None:
                            row = idx.row()
                            self.items.insert(row + 2, self.items[row])
                            self.items.pop(row)
                            self.QList.clear()
                            self.QList.addItems(self.items)
            except Exception as e:
                print(e)
        else:
            print('Select at least one item from list!')
        pass


    def btn_new_clicked(self):
        self.module.roi_creator.start(new=True)

    def btn_edit_clicked(self):
        self.module.roi_creator.start(False)

    def btn_del_clicked(self):
        indexes = self.QList.selectedIndexes()
        if len(indexes) <= 0:
            print('Select at least one item from list!')
            QtWidgets.QMessageBox.about(self, "", "Select at least one item from list!")
            return
        reply = QtWidgets.QMessageBox.question(None, "Delete", "Are you sure to delete {}?".format(
            self.QList.currentItem().text()), QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            print(indexes)
            for idx in indexes[::-1]:
                self.items.pop(idx)
            self.QList.clear()
            self.QList.addItems(self.items)



