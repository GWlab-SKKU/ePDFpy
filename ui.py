from PyQt5 import QtCore, QtWidgets, QtGui
import os, sys
import pyqtgraph as pg
import file
import numpy as np
import image_process


class DataViewer(QtWidgets.QMainWindow):
    def __init__(self, argv):
        self.qtapp = QtWidgets.QApplication.instance()
        if not self.qtapp:
            self.qtapp = QtWidgets.QApplication(argv)
        QtWidgets.QMainWindow.__init__(self)
        self.this_dir, self.this_filename = os.path.split(__file__)

        # Make settings collection

        self.strain_window = None
        self.current_files = []
        self.current_files_num = 0

        self.main_window = QtWidgets.QWidget()
        self.main_window.setWindowTitle("pdf_tools")

        self.controlPanel = ControlPanel()
        self.controlPanel.setMaximumWidth(300)
        self.imgPanel = ImgPanel()
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.controlPanel)
        layout.addWidget(self.imgPanel)
        self.main_window.setLayout(layout)
        self.btn_binding()

        self.main_window.show()

    def btn_binding(self):
        self.controlPanel.openFilePanel.btn_path.clicked.connect(self.open_file_path)
        self.controlPanel.operationPanel.btn_find_center.clicked.connect(self.find_center)
        self.imgPanel.btn_left.clicked.connect(self.btn_left_clicked)
        self.imgPanel.btn_right.clicked.connect(self.btn_right_clicked)

    def find_center(self):
        i1 = int(self.controlPanel.settingPanel.text_intensity_range_num1.toPlainText())
        i2 = int(self.controlPanel.settingPanel.text_intensity_range_num2.toPlainText())
        intensity_range = (i1,i2)
        slice_num = int(self.controlPanel.settingPanel.text_slice_num.toPlainText())

        center = image_process.get_center(self.img,intensity_range,slice_num)
        image_process.draw_center_line(self.img, center)
        self.imgPanel.update_img(self.img)

    def open_file_path(self):
        self.current_files.clear()
        if self.controlPanel.openFilePanel.radio_file.isChecked():
            path,_ = QtWidgets.QFileDialog.getOpenFileNames(self,'open')
            if len(path)==0:
                return
            self.current_files.extend(path)

        elif self.controlPanel.openFilePanel.radio_folder.isChecked():
            path = QtWidgets.QFileDialog.getExistingDirectory(self,'open')
            self.current_files.extend(file.get_file_list_from_path(path,'.mrc'))

        else:
            print("ERROR with open File")
            return

        self.read_img(0)

        self.controlPanel.openFilePanel.lbl_path.setText(str(self.current_files))
        self.controlPanel.openFilePanel.lbl_file_count_num.setText(str(len(self.current_files)))

    def btn_right_clicked(self):
        if not self.current_files_num==len(self.current_files)-1:
            self.read_img(self.current_files_num+1)

    def btn_left_clicked(self):
        if not self.current_files_num == 0:
            self.read_img(self.current_files_num-1)

    def read_img(self,i):
        self.current_files_num = i
        self.raw, self.img = file.load_mrc_img(self.current_files[self.current_files_num])
        self.imgPanel.update_img(self.img)
        self.imgPanel.lbl_current_num.setText(str(i+1)+"/"+str(len(self.current_files)))


class ControlPanel(QtWidgets.QWidget):
    text_fixed_height = 25
    text_fixed_width = 70

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.openFilePanel = self.OpenFilePanel()
        self.settingPanel = self.SettingPanel()
        self.operationPanel = self.OperationPanel()

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.openFilePanel)
        layout.addWidget(self.settingPanel)
        layout.addWidget(self.operationPanel)
        layout.addStretch(1)
        self.setLayout(layout)

    class OpenFilePanel(QtWidgets.QWidget):
        def __init__(self):
            QtWidgets.QWidget.__init__(self)
            layout = QtWidgets.QGridLayout()

            grp_1 = QtWidgets.QGroupBox()
            self.radio_folder = QtWidgets.QRadioButton("Folder")
            self.radio_file = QtWidgets.QRadioButton("File")
            grp_1_layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)
            grp_1.setLayout(grp_1_layout)
            grp_1_layout.addWidget(self.radio_folder)
            grp_1_layout.addWidget(self.radio_file)
            self.radio_file.setChecked(True)

            grp_2 = QtWidgets.QGroupBox()
            grp_2_layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)

            lbl_file_count = QtWidgets.QLabel("filecount:")
            self.lbl_file_count_num = QtWidgets.QLabel("0")
            grp_2_layout.addWidget(lbl_file_count)
            grp_2_layout.addWidget(self.lbl_file_count_num)
            grp_2.setLayout(grp_2_layout)

            self.lbl_path = QtWidgets.QLabel("file_path")
            self.btn_path = QtWidgets.QPushButton("open")
            self.lbl_path.setFixedHeight(ControlPanel.text_fixed_height)
            self.lbl_path.setMaximumWidth(300)
            grp_3 = QtWidgets.QGroupBox()
            grp_3_layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)
            grp_3_layout.addWidget(self.lbl_path)
            grp_3_layout.addWidget(self.btn_path)
            grp_3.setLayout(grp_3_layout)

            layout.addWidget(grp_1, 0, 0)
            layout.addWidget(grp_2, 0, 1)
            layout.addWidget(grp_3, 1, 0, 1, 2)
            self.setLayout(layout)

    class SettingPanel(QtWidgets.QWidget):
        def __init__(self):
            QtWidgets.QWidget.__init__(self)
            layout = QtWidgets.QGridLayout()

            lbl_intensity_range = QtWidgets.QLabel("intensity range(255)")
            lbl_slice_num = QtWidgets.QLabel("slice number")
            self.text_intensity_range_num1 = QtWidgets.QTextEdit('130')
            self.text_intensity_range_num2 = QtWidgets.QTextEdit('135')
            self.text_slice_num = QtWidgets.QTextEdit('1')
            self.text_intensity_range_num1.setFixedHeight(ControlPanel.text_fixed_height)
            self.text_intensity_range_num2.setFixedHeight(ControlPanel.text_fixed_height)
            self.text_slice_num.setFixedHeight(ControlPanel.text_fixed_height)
            self.text_intensity_range_num1.setFixedWidth(ControlPanel.text_fixed_width)
            self.text_intensity_range_num2.setFixedWidth(ControlPanel.text_fixed_width)
            self.text_slice_num.setFixedWidth(ControlPanel.text_fixed_width)

            grp_1 = QtWidgets.QGroupBox("View Mode")
            self.radio_viewmode_raw = QtWidgets.QRadioButton("raw")
            self.radio_viewmode_log = QtWidgets.QRadioButton("log")
            self.radio_viewmode_colorcube = QtWidgets.QRadioButton("colorcube")
            grp_1_layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)
            grp_1_layout.addWidget(self.radio_viewmode_raw)
            grp_1_layout.addWidget(self.radio_viewmode_log)
            grp_1_layout.addWidget(self.radio_viewmode_colorcube)
            grp_1.setLayout(grp_1_layout)
            self.radio_viewmode_log.setChecked(True)

            layout.addWidget(grp_1,0,0,1,4)
            layout.addWidget(lbl_intensity_range, 1, 0, 1, 2)
            layout.addWidget(lbl_slice_num, 2, 0, 1, 2)
            layout.addWidget(self.text_intensity_range_num1, 1, 2)
            layout.addWidget(self.text_intensity_range_num2, 1, 3)
            layout.addWidget(self.text_slice_num, 2, 2)

            self.setLayout(layout)

    class OperationPanel(QtWidgets.QWidget):
        def __init__(self):
            QtWidgets.QWidget.__init__(self)
            layout = QtWidgets.QGridLayout()
            self.btn_find_center = QtWidgets.QPushButton("find center")
            self.btn_get_azimuthal_avg = QtWidgets.QPushButton("get azimuthal data")
            self.btn_autostart = QtWidgets.QPushButton("auto start & save")

            layout.addWidget(self.btn_find_center,0,0)
            layout.addWidget(self.btn_get_azimuthal_avg, 0, 1)
            layout.addWidget(self.btn_autostart, 1, 0,1,2)

            self.setLayout(layout)

class ImgPanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.imageView = pg.ImageView()
        self.btn_left = QtWidgets.QPushButton("<<")
        self.btn_right = QtWidgets.QPushButton(">>")
        self.lbl_current_num = QtWidgets.QLabel("1")
        self.lbl_current_num.setAlignment(QtCore.Qt.AlignCenter)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.imageView,0,0,1,9)
        layout.addWidget(self.btn_left,1,0,1,4)
        layout.addWidget(self.btn_right,1,5,1,4)
        layout.addWidget(self.lbl_current_num, 1, 4)
        self.setMinimumWidth(500)
        self.setMinimumHeight(500)
        self.setLayout(layout)
        self._current_data = None
    def update_img(self,img):
        cmap = pg.ColorMap(np.linspace(0,1,len(image_process.colorcube)),color=image_process.colorcube)
        self.imageView.setColorMap(cmap)
        self._current_data=img
        if len(img.shape) == 2:
            self.imageView.setImage(self._current_data.transpose(1,0))
        if len(img.shape) == 3:
            self.imageView.setImage(self._current_data.transpose(1,0,2))
    def get_img(self):
        return self._current_data


if __name__ == '__main__':
    app = DataViewer(sys.argv)
    sys.exit(app.qtapp.exec_())
