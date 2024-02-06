import pyqtgraph as pg
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import cv2
import os
from epdfpy.file import load
from epdfpy import definitions
from epdfpy.calculate import beam_stopper
from PyQt5 import QtCore
import json
import sys


class MaskModule(QtCore.QObject):
    mask_changed = QtCore.pyqtSignal()  # to update mask image
    data_changed = QtCore.pyqtSignal()

    def __init__(self, img = None, imageView = None, fp = None):
        super(MaskModule, self).__init__()
        self.img = img
        self.fp = fp
        self.imageView = imageView
        self.mask_dict = self.mask_load()

        self.roi_creator: RoiCreater = None
        self.dropdown = DropDown(self)
        self.list_widget = ListWidget(self)
        self.mask = None

        self._mask_reload()

    def update_img(self, img):
        self.img = img
        self.dropdown.img = img
        self.list_widget.img = img

    def mask_save(self):
        if not os.path.isdir(os.path.split(self.fp)[0]):
            return
        if self.fp is None:
            return
        if len(self.mask_dict) == 0:
            return
        save_dict = {}
        for key, value in self.mask_dict.copy().items():
            temp = value.copy()
            temp.update({'data':temp['data'].tolist()})
            save_dict.update({key: temp})
        with open(self.fp, 'w') as json_file:
            json.dump(save_dict, json_file)

    def mask_load(self):
        try:
            with open(self.fp, 'r') as json_file:
                self.mask_dict = json.load(json_file)
                for key, value in self.mask_dict.items():
                    value.update({'data': np.array(value['data'])})
                    self.mask_dict.update({key: value})
            return self.mask_dict
        except Exception as e:
            print("failed to load ",self.fp, e)
            return {}

    def get_current_mask(self):
        if self.dropdown.currentText() in ['None', "[Edit]"]:
            # self.mask = np.loadtxt(definitions.MASK_PATH_DEFAULT,delimiter=',').astype(np.uint8)
            self.mask = None
        else:
            x_y = self.mask_dict[self.dropdown.currentText()]['data']
            img = np.zeros(self.img.shape, dtype=np.uint8)
            cv2.fillPoly(img, pts=[x_y], color=(255, 255, 255))
            self.mask = img
        return self.mask

    def _mask_reload(self):
        self.dropdown.mask_reload()
        self.list_widget.mask_reload()

class RoiCreater(QtWidgets.QMainWindow):
    def __init__(self, module:MaskModule, img, name=None, pnts=None):
        QtWidgets.QWidget.__init__(self)
        self.module = module
        self.img = img
        self.name = name

        self.init_ui()

        self.initial_image_load(self.img)

        self.create_menubar()

        self.binding()

        self.setWindowTitle("Masking")

    def initial_image_load(self, img):
        self.update_image(img)
        handles = self.get_handles()
        if handles is None:
            self.draw_roi()
        # else:
        #     self.draw_roi(handles)

        if self.name is not None:
            self.txt_name.setText(self.name)

    def init_ui(self):
        # self.setWindowFlags(self.windowFlags() | Qt.Qt.Window)
        layout = QtWidgets.QVBoxLayout()
        control_layout = QtWidgets.QHBoxLayout()
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

        control_layout.addWidget(grp_view_mode)
        control_layout.addWidget(grp_save)
        layout.addWidget(self.imageView)
        layout.addLayout(control_layout)
        self.mainWidget = QtWidgets.QWidget()
        self.mainWidget.setLayout(layout)
        self.setCentralWidget(self.mainWidget)

        # self.setBaseSize(800,800)
        self.setMinimumSize(800, 700)

    def create_menubar(self):
        menubar = self.menuBar()
        menu_file = menubar.addMenu("\tFile\t")

        self.action_import_image = QtWidgets.QAction("Import image", self)
        self.action_import_stem_image = QtWidgets.QAction("Import stem image", self)
        self.action_import_poly = QtWidgets.QAction("Import polygon", self)
        self.action_export_poly = QtWidgets.QAction("Export polygon", self)
        self.action_export_mask = QtWidgets.QAction("Export mask", self)

        menu_file.addAction(self.action_import_image)
        menu_file.addAction(self.action_import_stem_image)
        menu_file.addAction(self.action_import_poly)
        menu_file.addAction(self.action_export_poly)
        menu_file.addAction(self.action_export_mask)
        return menubar

    def update_image(self, img=None):
        if img is None:
            img = self.module.img
        if img is None:
            img = self.img
        if img is None:
            return
        self.img = img
        if self.radio_raw.isChecked():
            disp_img = img
        if self.radio_root.isChecked():
            disp_img = np.power(img, 0.5)
        if self.radio_log.isChecked():
            disp_img = np.log((np.abs(img)+1))
        self.imageView.setImage(disp_img.T)

    def draw_roi(self, pnts=None):
        if hasattr(self,'poly_line_roi'):
            pass  # todo:

        if self.img is None:
            return
        if pnts is None:
            pnts = beam_stopper.find_polygon(self.img)
            if pnts is not None:
                pnts = pnts[:,0,:]
                # pnts = np.flip(pnts, axis=None)
        if pnts is None:
            w = self.img.shape[0]
            h = self.img.shape[1]
            pnts = [[int(w / 2) - int(w / 10), int(h / 2) - int(h / 10)],
                    [int(w / 2) + int(w / 10), int(h / 2) - int(h / 10)],
                    [int(w / 2) + int(w / 10), int(h / 2) + int(h / 10)],
                    [int(w / 2) - int(w / 10), int(h / 2) + int(h / 10)]]

        self.draw_poly(pnts)

    def draw_poly(self, pnts):
        if hasattr(self,'poly_line_roi') and isinstance(self.poly_line_roi,pg.ROI):
            pnts = pnts - self.poly_line_roi.pos()
            self.poly_line_roi.setPoints(pnts, closed=True)
        else:
            self.poly_line_roi = pg.PolyLineROI(pnts, closed=True)
            self.imageView.addItem(self.poly_line_roi)

    def binding(self):
        self.btn_ok.clicked.connect(self.btn_ok_clicked)
        self.btn_cancel.clicked.connect(self.btn_cancel_clicked)
        self.radio_raw.toggled.connect(lambda x: self.update_image())
        self.radio_root.toggled.connect(lambda x: self.update_image())
        self.radio_log.toggled.connect(lambda x: self.update_image())
        self.action_import_image.triggered.connect(self.import_image)
        self.action_import_stem_image.triggered.connect(self.import_stem_image)
        self.action_import_poly.triggered.connect(self.import_poly)
        self.action_export_mask.triggered.connect(self.export_mask)
        self.action_export_poly.triggered.connect(self.export_poly)

    def import_image(self):
        fp, _ = QFileDialog.getOpenFileName()
        if fp:
            img = load.load_diffraction_image(fp)
            self.initial_image_load(img)
            print(f"diffraction image is imported, {fp}")

    def import_stem_image(self):
        fp, _ = QFileDialog.getOpenFileName()
        if fp:
            img = load.load_stem_image(fp)
            reply = QtWidgets.QMessageBox.question(None, "","Use variance for representative image?", QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                img = np.var(img,axis=0)
            else:
                img = np.mean(img, axis=0)
            self.initial_image_load(img)
            print(f"stem image is imported, {fp}")


    def import_poly(self):
        if self.img is None:
            QtWidgets.QMessageBox.about(None, "You have to load image first.")
            return
        fp, _ = QFileDialog.getOpenFileName()
        if fp:
            pnts = np.loadtxt(fp, delimiter=',')
            self.draw_poly(pnts)
            print(f"poly is imported, {fp}")

    def get_handles(self):
        if not hasattr(self,'poly_line_roi'):
            return None
        else:
            handles = [handle.pos() for handle in self.poly_line_roi.getHandles()]
            handles = np.array(handles) + np.array(self.poly_line_roi.pos())
            handles = handles.astype(int)
            self.draw_poly(handles)
            return handles

    def export_mask(self):
        handles = self.get_handles()
        if handles is None:
            QtWidgets.QMessageBox.about(self,"Error","No pnts are available now")
            return

        img = np.zeros(self.imageView.image.shape)
        cv2.fillPoly(img, pts=[handles], color=(255, 255, 255))

        fp, _ = QFileDialog.getSaveFileName()
        if fp == "":
            return

        if os.path.splitext(fp)[1] is None or os.path.splitext(fp)[1] != ".csv":
            fp = fp + ".csv"
        np.savetxt(fp, img, delimiter=',', fmt='%s')
        print("save to {}".format(fp))

    def export_poly(self):
        handles = self.get_handles()
        if handles is None:
            QtWidgets.QMessageBox.about(self, "Error", "No pnts are available now")
            return

        fp, _ = QFileDialog.getSaveFileName()
        if fp == "":
            return

        if os.path.splitext(fp)[1] is None or os.path.splitext(fp)[1] != ".csv":
            fp = fp + ".csv"

        np.savetxt(fp, handles, delimiter=',', fmt='%s')
        print("save to {}".format(fp))


    def btn_ok_clicked(self):
        if self.txt_name.toPlainText() == "":
            QtWidgets.QMessageBox.about(self, "", "Enter the mask name")
            return
        handles = self.get_handles()
        if handles is None:
            QtWidgets.QMessageBox.about(self, "Error", "No pnts are available now")
            return
        self.module.mask_dict.update(
            {self.txt_name.toPlainText():
                 {'size':self.module.img.shape,
                  'data':handles}
             })
        self.module._mask_reload()
        self.module.get_current_mask()
        self.module.mask_changed.emit()
        self.close()

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
        if self.currentText() not in ['[Edit]','None', ''] and self.img is not None:
            size = self.module.mask_dict[self.currentText()]['size']
            shape = self.img.shape
            if tuple(size) != tuple(shape):
                QtWidgets.QMessageBox.about(self,"",
                        f"Size is not matching. \n mask size: {size}\n image size: {shape}")
                return
        self.module.get_current_mask()
        self.module.mask_changed.emit()

class ListWidget(QtWidgets.QWidget):
    def __init__(self, module:MaskModule):
        super().__init__()
        self.module = module
        self.QList = QtWidgets.QListWidget()
        self.mask_reload()
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

    def mask_reload(self):
        self.QList.clear()
        self.items = list(self.module.mask_dict.keys())
        self.QList.addItems(self.items)

    def btn_move_up_clicked(self):
        indexes = self.QList.selectedIndexes()
        if len(indexes) <= 0:
            print('Select at least one item from list!')
            return
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
        self.update_list_to_dropdown()

    def btn_move_down_clicked(self):
        max_row = len(self.items)
        indexes = self.QList.selectedIndexes()
        if len(indexes) <= 0:
            print('Select at least one item from list!')
            return
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
        self.update_list_to_dropdown()

    def update_list_to_dropdown(self):
        self.QList.update()
        new_mask_dict = {}
        if self.module is not None:
            for i in range(len(self.QList)):
                name = self.QList.item(i).text()
                new_mask_dict.update({name: self.module.mask_dict[name]})
            self.module.mask_dict = new_mask_dict
            self.module._mask_reload()

    def btn_new_clicked(self):
        self.module.roi_creator = RoiCreater(self.module, self.module.img)
        self.module.roi_creator.show()

    def btn_edit_clicked(self):
        self.module.roi_creator = RoiCreater(self.module, self.module.img,
                                             self.QList.selectedItems()[0].text(),
                                             self.module.mask_dict[self.QList.selectedItems()[0].text()]['data'])
        self.module.roi_creator.show()

    def btn_del_clicked(self):
        indexes = self.QList.selectedIndexes()
        if len(indexes) <= 0:
            print('Select at least one item from list!')
            QtWidgets.QMessageBox.about(self, "", "Select at least one item from list!")
            return
        reply = QtWidgets.QMessageBox.question(None, "Delete", "Are you sure to delete {}?".format(
            self.QList.currentItem().text()), QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.No:
            return
        for idx in indexes[::-1]:
            key = self.items.pop(idx.row())
            self.module.mask_dict.pop(key)
        self.QList.clear()
        self.QList.addItems(self.items)
        self.module.mask_changed.emit()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        super().closeEvent(a0)
        self.module.mask_save()
        self.module.get_current_mask()
        self.module.mask_changed.emit()


if __name__ == '__main__':
    qtapp = QtWidgets.QApplication.instance()
    if not qtapp:
        qtapp = QtWidgets.QApplication(sys.argv)
    maskModule = MaskModule(fp=definitions.MASK_PATH)

    app = maskModule.list_widget
    app.btn_new_clicked()

    sys.exit(qtapp.exec_())
