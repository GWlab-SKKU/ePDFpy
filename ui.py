from PyQt5 import QtCore, QtWidgets, QtGui
import os, sys
import pyqtgraph as pg
import file
import numpy as np
import image_process
import util


class DataViewer(QtWidgets.QMainWindow):
    def __init__(self, argv):
        self.qtapp = QtWidgets.QApplication.instance()
        if not self.qtapp:
            self.qtapp = QtWidgets.QApplication(argv)
        QtWidgets.QMainWindow.__init__(self)
        self.this_dir, self.this_filename = os.path.split(__file__)

        # Make settings collection

        self.main_window = MainWindow()
        self.main_window.show()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):

        self.plotWindow = None
        self.current_files = []
        self.current_page = 0

        QtWidgets.QWidget.__init__(self)
        self.controlPanel = ControlPanel()
        self.controlPanel.setMaximumWidth(330)
        self.imgPanel = ImgPanel()
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.controlPanel)
        self.layout.addWidget(self.imgPanel)
        self.setLayout(self.layout)
        self.btn_binding()
        self.isShowCenter=True
        self.resize(1080,600)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        util.settings["intensity_range_1"] = self.controlPanel.settingPanel.spinBox_irange1.value()
        util.settings["intensity_range_2"] = self.controlPanel.settingPanel.spinBox_irange2.value()
        util.settings["slice_number"] = self.controlPanel.settingPanel.spinBox_slice_num.value()
        util.settings["show_center_line"] = self.controlPanel.settingPanel.chkBox_show_centerLine.isChecked()
        util.save_settings(util.settings)

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() == QtCore.Qt.Key.Key_Right:
            self.controlPanel.settingPanel.spinBox_center_x.setValue(
                self.controlPanel.settingPanel.spinBox_center_x.value()+1
            )
        if e.key() == QtCore.Qt.Key.Key_Left:
            self.controlPanel.settingPanel.spinBox_center_x.setValue(
                self.controlPanel.settingPanel.spinBox_center_x.value()-1
            )
        if e.key() == QtCore.Qt.Key.Key_Up:
            self.controlPanel.settingPanel.spinBox_center_y.setValue(
                self.controlPanel.settingPanel.spinBox_center_y.value()-1
            )
        if e.key() == QtCore.Qt.Key.Key_Down:
            self.controlPanel.settingPanel.spinBox_center_y.setValue(
                self.controlPanel.settingPanel.spinBox_center_y.value()+1
            )
        if e.key() == QtCore.Qt.Key.Key_PageUp:
            self.btn_left_clicked()
        if e.key() == QtCore.Qt.Key.Key_PageDown:
            self.btn_right_clicked()


    def btn_binding(self):
        self.controlPanel.openFilePanel.btn_path.clicked.connect(self.open_file_path)
        self.controlPanel.operationPanel.btn_find_center.clicked.connect(lambda: (self.find_center(),self.draw_center()))
        self.imgPanel.btn_left.clicked.connect(self.btn_left_clicked)
        self.imgPanel.btn_right.clicked.connect(self.btn_right_clicked)
        self.controlPanel.operationPanel.btn_get_azimuthal_avg.clicked.connect(self.get_azimuthal_value)
        self.controlPanel.settingPanel.spinBox_center_x.valueChanged.connect(self.draw_center)
        self.controlPanel.settingPanel.spinBox_center_y.valueChanged.connect(self.draw_center)
        self.controlPanel.operationPanel.btn_save_current_azimuthal.clicked.connect(self.save_current_azimuthal)
        self.controlPanel.operationPanel.btn_save_all_azimuthal.clicked.connect(self.save_all_azimuthal)
        self.controlPanel.settingPanel.chkBox_show_centerLine.stateChanged.connect(self.show_centerLine)

    def show_centerLine(self):
        if self.controlPanel.settingPanel.chkBox_show_centerLine.isChecked():
            self.draw_center()
        else:
            self.imgPanel.update_img(self.img)

    def is_center_ready(self, i=None):
        if not hasattr(self,'center'):
            return False
        if i is not None:
            page_num = i
        else:
            page_num = self.current_page
        return self.center[page_num].sum() != 0 and self.center[page_num].sum() != 2

    def save_current_azimuthal(self):
        if not self.is_center_ready():
            self.find_center()
            self.draw_center()
        if self.azavg is None:
            self.get_azimuthal_value()
        file.save_current_azimuthal(self.azavg, self.current_files[self.current_page], True)
        file.save_current_azimuthal(self.azvar, self.current_files[self.current_page], False)
        folder_path, file_full_name = os.path.split(self.current_files[self.current_page])
        file_name, ext = os.path.splitext(file_full_name)
        img_file_path = os.path.join(folder_path, file.analysis_folder_name, file_name+"_img.tiff")
        self.imgPanel.imageView.export(img_file_path)

    def save_all_azimuthal(self):
        for i in range(len(self.current_files)):
            print("processing auto_save azimuthal values", self.current_files)
            self.read_img(i)
            self.save_current_azimuthal()
            self.controlPanel.operationPanel.progress_bar.setValue(i+1/len(self.current_files))
        self.controlPanel.operationPanel.progress_bar.setValue(0)

    def get_azimuthal_value(self):
        self.azavg, self.azvar = image_process.get_azimuthal_average(self.raw, self.center[self.current_page])
        if self.plotWindow is None:
            self.plotWindow = QtWidgets.QWidget()
            self.plotWindow.layout = QtWidgets.QHBoxLayout()
            self.plotWindow.layout.setSpacing(0)
            self.plotWindow.layout.setContentsMargins(0,0,0,0)
            self.plotWidget1 = pg.PlotWidget(title='average')
            self.plotWidget2 = pg.PlotWidget(title='variance')
            self.plotWindow.layout.addWidget(self.plotWidget1)
            self.plotWindow.layout.addWidget(self.plotWidget2)
            self.plotWindow.setLayout(self.plotWindow.layout)
            self.plotWidget1.plot(self.azavg,pen=(255,0,0))
            self.plotWidget2.plot(self.azvar,pen=(0,255,0))
            self.plotWindow.resize(700,350)
            self.plotWindow.show()
        else:
            self.plotWidget1.clear()
            self.plotWidget1.plot(self.azavg,pen=(255,0,0))
            self.plotWidget2.clear()
            self.plotWidget2.plot(self.azvar,pen=(0,255,0))

    # def shift_center(self, x, y):
    #     if not hasattr(self,'center'):
    #         return
    #     self.center[self.current_page][0] += x
    #     self.center[self.current_page][1] += y
    #     print("center moved! ", self.center[self.current_page])
    #     self.draw_center()

    def find_center(self):
        i1 = self.controlPanel.settingPanel.spinBox_irange1.value()
        i2 = self.controlPanel.settingPanel.spinBox_irange2.value()
        intensity_range = (i1,i2)
        slice_num = int(self.controlPanel.settingPanel.spinBox_slice_num.value())
        self.center[self.current_page] = image_process.get_center(self.img,intensity_range,slice_num)
        self.put_center_to_spinBoxes()
        # you must use self.draw_center() after find_center
        return self.center[self.current_page]

    def draw_center(self):
        if not hasattr(self, 'center') or not self.controlPanel.settingPanel.chkBox_show_centerLine.isChecked():
            return
        self.center[self.current_page][0] = self.controlPanel.settingPanel.spinBox_center_x.value()
        self.center[self.current_page][1] = self.controlPanel.settingPanel.spinBox_center_y.value()
        lined_img = self.img.copy()
        image_process.draw_center_line(lined_img, self.center[self.current_page])
        self.imgPanel.update_img(lined_img)

    def open_file_path(self):
        self.current_files.clear()
        if self.controlPanel.openFilePanel.radio_file.isChecked():
            path,_ = QtWidgets.QFileDialog.getOpenFileNames(self,'open')
            if len(path)==0:
                return
            self.current_files.extend(path)

        elif self.controlPanel.openFilePanel.radio_folder.isChecked():
            path = QtWidgets.QFileDialog.getExistingDirectory(self,'open')
            if len(path)==0:
                return
            self.current_files.extend(file.get_file_list_from_path(path,'.mrc'))
        else:
            print("ERROR with open File")
            return

        self.read_img(0)
        self.center = np.zeros((len(self.current_files), 2))
        self.controlPanel.openFilePanel.lbl_path.setText(str(self.current_files))
        self.controlPanel.settingPanel.spinBox_center_x.setMaximum(self.imgPanel.imageView.image.shape[1])
        self.controlPanel.settingPanel.spinBox_center_y.setMaximum(self.imgPanel.imageView.image.shape[0])

    def btn_right_clicked(self):
        if not self.current_page == len(self.current_files) - 1:
            self.read_img(self.current_page + 1)

    def btn_left_clicked(self):
        if not self.current_page == 0:
            self.read_img(self.current_page - 1)

    def read_img(self,i):
        self.current_page = i
        self.azavg = None
        self.azvar = None
        self.put_center_to_spinBoxes()
        self.raw, self.img = file.load_mrc_img(self.current_files[self.current_page])
        if self.is_center_ready():
            self.draw_center()
        else:
            self.imgPanel.update_img(self.img)
        self.imgPanel.lbl_current_num.setText(str(i+1)+"/"+str(len(self.current_files)))
        self.setWindowTitle(self.current_files[self.current_page])

    def put_center_to_spinBoxes(self, center=None):
        if not hasattr(self, 'center'):
            return
        self.controlPanel.settingPanel.spinBox_center_x.valueChanged.disconnect()
        self.controlPanel.settingPanel.spinBox_center_y.valueChanged.disconnect()
        if center is not None:
            self.center[self.current_page] = center
        self.controlPanel.settingPanel.spinBox_center_x.setValue(self.center[self.current_page][0])
        self.controlPanel.settingPanel.spinBox_center_y.setValue(self.center[self.current_page][1])
        self.controlPanel.settingPanel.spinBox_center_x.valueChanged.connect(self.draw_center)
        self.controlPanel.settingPanel.spinBox_center_y.valueChanged.connect(self.draw_center)


class ControlPanel(QtWidgets.QWidget):
    # text_fixed_height = 25
    # text_fixed_width = 70

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.openFilePanel = self.OpenFilePanel("OpenFile")
        self.settingPanel = self.SettingPanel("Settings")
        self.operationPanel = self.OperationPanel("Operation")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.openFilePanel)
        layout.addWidget(self.settingPanel)
        layout.addWidget(self.operationPanel)
        layout.addStretch(1)
        self.setLayout(layout)

    class OpenFilePanel(QtWidgets.QGroupBox):
        def __init__(self,arg):
            QtWidgets.QGroupBox.__init__(self,arg)
            layout = QtWidgets.QGridLayout()

            radio_grp = QtWidgets.QGroupBox()
            self.radio_folder = QtWidgets.QRadioButton("Folder")
            self.radio_file = QtWidgets.QRadioButton("File")
            radio_grp_layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)
            radio_grp.setLayout(radio_grp_layout)
            radio_grp_layout.addWidget(self.radio_folder)
            radio_grp_layout.addWidget(self.radio_file)
            self.radio_file.setChecked(True)

            self.lbl_path = QtWidgets.QLabel("/")
            self.btn_path = QtWidgets.QPushButton("open")
            # self.lbl_path.setFixedHeight(ControlPanel.text_fixed_height)
            self.lbl_path.setMaximumWidth(300)


            layout.addWidget(radio_grp, 0, 0)
            layout.addWidget(self.btn_path, 0, 1)
            layout.addWidget(self.lbl_path, 1, 0, 1, 2)
            self.setLayout(layout)

    class SettingPanel(QtWidgets.QGroupBox):
        def __init__(self,arg):
            QtWidgets.QGroupBox.__init__(self,arg)
            layout = QtWidgets.QGridLayout()

            lbl_intensity_range = QtWidgets.QLabel("intensity range(255)")
            lbl_slice_num = QtWidgets.QLabel("slice number")
            lbl_center = QtWidgets.QLabel("center")
            self.spinBox_irange1 = QtWidgets.QSpinBox()
            self.spinBox_irange2 = QtWidgets.QSpinBox()
            self.spinBox_slice_num = QtWidgets.QSpinBox()
            self.spinBox_center_x = QtWidgets.QSpinBox()
            self.spinBox_center_y = QtWidgets.QSpinBox()
            self.chkBox_show_centerLine = QtWidgets.QCheckBox("Show center line")
            self.spinBox_irange1.setMinimum(1)
            self.spinBox_irange2.setMinimum(1)
            self.spinBox_slice_num.setMinimum(1)
            self.spinBox_center_x.setMinimum(1)
            self.spinBox_center_y.setMinimum(1)
            self.spinBox_irange1.setMaximum(255)
            self.spinBox_irange2.setMaximum(255)
            self.spinBox_slice_num.setMaximum(255)
            self.spinBox_irange1.setValue(util.settings["intensity_range_1"])
            self.spinBox_irange2.setValue(util.settings["intensity_range_2"])
            self.spinBox_slice_num.setValue(util.settings["slice_number"])
            self.chkBox_show_centerLine.setChecked(util.settings["show_center_line"])
            # self.spinBox_irange1.setFixedHeight(ControlPanel.text_fixed_height)
            # self.spinBox_irange2.setFixedHeight(ControlPanel.text_fixed_height)
            # self.spinBox_slice_num.setFixedHeight(ControlPanel.text_fixed_height)
            # self.spinBox_irange1.setFixedWidth(ControlPanel.text_fixed_width)
            # self.spinBox_irange2.setFixedWidth(ControlPanel.text_fixed_width)
            # self.spinBox_slice_num.setFixedWidth(ControlPanel.text_fixed_width)

            # grp_1 = QtWidgets.QGroupBox("View Mode")
            # self.radio_viewmode_raw = QtWidgets.QRadioButton("raw")
            # self.radio_viewmode_log = QtWidgets.QRadioButton("log")
            # self.radio_viewmode_colorcube = QtWidgets.QRadioButton("colorcube")
            # grp_1_layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)
            # grp_1_layout.addWidget(self.radio_viewmode_raw)
            # grp_1_layout.addWidget(self.radio_viewmode_log)
            # grp_1_layout.addWidget(self.radio_viewmode_colorcube)
            # grp_1.setLayout(grp_1_layout)

            # self.radio_viewmode_log.setChecked(True)

            # layout.addWidget(grp_1,0,0,1,4)
            layout.addWidget(lbl_intensity_range, 1, 0, 1, 2)
            layout.addWidget(lbl_slice_num, 2, 0, 1, 2)
            layout.addWidget(self.spinBox_irange1, 1, 2)
            layout.addWidget(self.spinBox_irange2, 1, 3)
            layout.addWidget(self.spinBox_slice_num, 2, 2)
            layout.addWidget(lbl_center,3,0,1,2)
            layout.addWidget(self.spinBox_center_x,3,2)
            layout.addWidget(self.spinBox_center_y, 3,3)
            layout.addWidget(self.chkBox_show_centerLine,4,0)

            self.setLayout(layout)

    class OperationPanel(QtWidgets.QGroupBox):
        def __init__(self,arg):
            QtWidgets.QGroupBox.__init__(self,arg)
            layout = QtWidgets.QGridLayout()
            self.btn_find_center = QtWidgets.QPushButton("find center")
            self.btn_get_azimuthal_avg = QtWidgets.QPushButton("get azimuthal data")
            self.btn_save_current_azimuthal = QtWidgets.QPushButton("save current azimuthal data")
            self.btn_save_all_azimuthal = QtWidgets.QPushButton("save every azimuthal data")
            self.progress_bar = QtWidgets.QProgressBar()
            self.progress_bar.setValue(0)

            layout.addWidget(self.btn_find_center, 0, 0)
            layout.addWidget(self.btn_get_azimuthal_avg, 0, 1)
            layout.addWidget(self.btn_save_current_azimuthal, 1, 0,1,2)
            layout.addWidget(self.btn_save_all_azimuthal, 2, 0,1,2)
            layout.addWidget(self.progress_bar,3,0,1,2)

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
        layout.addWidget(self.lbl_current_num,1,4)
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


