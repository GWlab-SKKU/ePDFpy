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
        QtWidgets.QWidget.__init__(self)
        self.plotWindow = None
        self.current_files = []
        self.current_page = 0
        self.upper = QtWidgets.QFrame(self)
        self.lower = QtWidgets.QFrame(self)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.controlPanel = ControlPanel()
        self.controlPanel.setMaximumWidth(330)
        self.imgPanel = ImgPanel()
        self.graphPanel = GraphPanel()
        self.upper_layout = QtWidgets.QHBoxLayout()
        self.upper_layout.addWidget(self.controlPanel)
        self.upper_layout.addWidget(self.imgPanel)
        self.upper.setLayout(self.upper_layout)
        self.lower_layout = QtWidgets.QHBoxLayout()
        self.lower_layout.addWidget(self.graphPanel)
        self.lower.setLayout(self.lower_layout)
        self.splitter.addWidget(self.upper)
        self.splitter.addWidget(self.lower)
        self.splitter.setStretchFactor(1,1)
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.splitter)
        self.lower_layout.setSpacing(0)
        self.lower_layout.setContentsMargins(0,0,0,0)
        self.upper_layout.setSpacing(0)
        self.upper_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.btn_binding()
        self.isShowCenter=True
        self.resize(1080,600)
        self.flag_range_update = False

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        util.settings["intensity_range_1"] = self.controlPanel.settingPanel.spinBox_irange1.value()
        util.settings["intensity_range_2"] = self.controlPanel.settingPanel.spinBox_irange2.value()
        util.settings["slice_count"] = self.controlPanel.settingPanel.spinBox_slice_count.value()
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
        self.controlPanel.operationPanel.btn_find_center.clicked.connect(lambda: (self.find_center(),self.update_img()))
        self.imgPanel.btn_left.clicked.connect(self.btn_left_clicked)
        self.imgPanel.btn_right.clicked.connect(self.btn_right_clicked)
        self.controlPanel.operationPanel.btn_get_azimuthal_avg.clicked.connect(self.get_azimuthal_value)
        self.controlPanel.settingPanel.spinBox_center_x.valueChanged.connect(self.update_img)
        self.controlPanel.settingPanel.spinBox_center_y.valueChanged.connect(self.update_img)
        self.controlPanel.operationPanel.btn_save_current_azimuthal.clicked.connect(self.save_current_azimuthal)
        self.controlPanel.operationPanel.btn_save_all_azimuthal.clicked.connect(self.save_all_azimuthal)
        self.controlPanel.settingPanel.chkBox_show_centerLine.stateChanged.connect(self.update_img)
        self.controlPanel.settingPanel.chkBox_show_beam_stopper_mask.stateChanged.connect(self.update_img)
        self.controlPanel.settingPanel.spinBox_pixel_range_left.valueChanged.connect(self.dialog_to_range)
        self.controlPanel.settingPanel.spinBox_pixel_range_right.valueChanged.connect(self.dialog_to_range)
        self.graphPanel.region.sigRegionChangeFinished.connect(self.range_to_dialog)
        self.graphPanel.button_start.clicked.connect(self.range_start_clicked)
        self.graphPanel.button_all.clicked.connect(self.range_all_clicked)
        self.graphPanel.button_end.clicked.connect(self.range_end_clicked)

    def range_start_clicked(self):
        left = self.controlPanel.settingPanel.spinBox_pixel_range_left.value()
        right = self.controlPanel.settingPanel.spinBox_pixel_range_right.value()
        l = left
        r = left+int((right-left)/4)
        print("left {}, right {}".format(l, r))
        mx = np.max(self.azavg[l:r])
        mn = np.min(self.azavg[l:r])
        self.graphPanel.plot_azav.setXRange(l, r, padding=0.1)
        self.graphPanel.plot_azav.setYRange(mn, mx, padding=0.1)
        print(self.graphPanel.plot_azav.viewRange())

    def range_all_clicked(self):
        l = self.controlPanel.settingPanel.spinBox_pixel_range_left.value()
        r = self.controlPanel.settingPanel.spinBox_pixel_range_right.value()
        mx = np.max(self.azavg[l:r])
        mn = np.min(self.azavg[l:r])
        self.graphPanel.plot_azav.setXRange(l, r, padding=0.1)
        self.graphPanel.plot_azav.setYRange(mn, mx, padding=0.1)

    def range_end_clicked(self):
        left = self.controlPanel.settingPanel.spinBox_pixel_range_left.value()
        right = self.controlPanel.settingPanel.spinBox_pixel_range_right.value()
        l = right-int((right - left) / 4)
        r = right
        mx = np.max(self.azavg[l:r])
        mn = np.min(self.azavg[l:r])
        self.graphPanel.plot_azav.setXRange(l, r, padding=0.1)
        self.graphPanel.plot_azav.setYRange(mn, mx, padding=0.1)



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
            self.update_img()
        if self.azavg is None:
            self.get_azimuthal_value()
        i_start_num = self.controlPanel.settingPanel.spinBox_irange1.value()
        i_end_num = self.controlPanel.settingPanel.spinBox_irange2.value()
        i_slice_num = self.controlPanel.settingPanel.spinBox_slice_count.value()
        i_list = [i_start_num,i_end_num,i_slice_num]
        file.save_current_azimuthal(self.azavg, self.current_files[self.current_page], True, i_slice=i_list)
        file.save_current_azimuthal(self.azvar, self.current_files[self.current_page], False, i_slice=i_list)
        folder_path, file_full_name = os.path.split(self.current_files[self.current_page])
        file_name, ext = os.path.splitext(file_full_name)
        img_file_path = os.path.join(folder_path, file.analysis_folder_name, file_name+"_img.tiff")
        self.imgPanel.imageView.export(img_file_path)

    def save_all_azimuthal(self):
        for i in range(len(self.current_files)):
            print("processing auto_save azimuthal values", self.current_files)
            self.read_img(i)
            self.save_current_azimuthal()
            self.controlPanel.operationPanel.progress_bar.setValue((i+1)/len(self.current_files))
        self.controlPanel.operationPanel.progress_bar.setValue(0)

    def get_azimuthal_value(self):
        # self.azavg, self.azvar = image_process.get_azimuthal_average(self.raw, self.center[self.current_page])
        self.azavg, self.azvar = image_process.get_azimuthal_average(self.img, self.center[self.current_page])
        self.graphPanel.update_graph(self.azavg)
        self.controlPanel.settingPanel.spinBox_pixel_range_right.setMaximum(len(self.azavg))
        self.controlPanel.settingPanel.spinBox_pixel_range_left.setMaximum(len(self.azavg))

        left = 0
        for i in range(len(self.azavg)):
            if int(self.azavg[i]) != 0 :
                left = i
                break
        self.graphPanel.region.setRegion([left, len(self.azavg)-1])
        # if self.plotWindow is None:
        #     self.plotWindow = QtWidgets.QWidget()
        #     self.plotWindow.layout = QtWidgets.QHBoxLayout()
        #     self.plotWindow.layout.setSpacing(0)
        #     self.plotWindow.layout.setContentsMargins(0,0,0,0)
        #     self.plot_azav = pg.PlotWidget(title='average')
        #     self.plot_azvar = pg.PlotWidget(title='variance')
        #     self.plotWindow.layout.addWidget(self.plot_azav)
        #     self.plotWindow.layout.addWidget(self.plot_azvar)
        #     self.plotWindow.setLayout(self.plotWindow.layout)
        #     self.plot_azav.plot(self.azavg, pen=(255, 0, 0))
        #     self.plot_azav.addItem(pg.LinearRegionItem([400, 700]))
        #     self.plot_azvar.plot(self.azvar, pen=(0, 255, 0))
        #     self.plotWindow.resize(1000,350)
        #     self.plotWindow.show()
        # else:
        #     self.plot_azav.clear()
        #     self.plot_azav.plot(self.azavg, pen=(255, 0, 0))
        #     self.plot_azvar.clear()
        #     self.plot_azvar.plot(self.azvar, pen=(0, 255, 0))

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
        slice_count = int(self.controlPanel.settingPanel.spinBox_slice_count.value())
        self.center[self.current_page] = image_process.get_center_gradient(self.img,intensity_range,slice_count)
        self.put_center_to_spinBoxes()
        # you must use self.draw_center() after find_center
        return self.center[self.current_page]

    def draw_center(self, img):
        self.center[self.current_page][0] = self.controlPanel.settingPanel.spinBox_center_x.value()
        self.center[self.current_page][1] = self.controlPanel.settingPanel.spinBox_center_y.value()
        lined_img = img.copy()
        image_process.draw_center_line(lined_img, self.center[self.current_page])
        return lined_img

    def open_file_path(self):
        # todo : check all file have same dimension, size
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
        self.controlPanel.settingPanel.spinBox_center_x.setMaximum(self.img.shape[1])
        self.controlPanel.settingPanel.spinBox_center_y.setMaximum(self.img.shape[0]) # todo : confusing x,y

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
        self.update_img()
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
        self.controlPanel.settingPanel.spinBox_center_x.valueChanged.connect(self.update_img)
        self.controlPanel.settingPanel.spinBox_center_y.valueChanged.connect(self.update_img)

    def update_img(self):
        if not hasattr(self,'img'):
            return
        img = self.img
        if self.is_center_ready() and self.controlPanel.settingPanel.chkBox_show_centerLine.isChecked():
            img = self.draw_center(img)
        if self.controlPanel.settingPanel.chkBox_show_beam_stopper_mask.isChecked():
            img = cv2.bitwise_and(img, img, mask=np.bitwise_not(image_process.mask))
        self.imgPanel.update_img(img)

    def dialog_to_range(self):
        self.flag_range_update = True
        left = self.controlPanel.settingPanel.spinBox_pixel_range_left.value()
        right = self.controlPanel.settingPanel.spinBox_pixel_range_right.value()
        self.graphPanel.region.setRegion([left,right])
        self.flag_range_update = False

    def range_to_dialog(self):
        if self.flag_range_update:
            return
        left, right = self.graphPanel.region.getRegion()
        left = np.round(left)
        right = np.round(right)
        print("range left",left,"rane right",right)
        self.graphPanel.region.disconnect()
        self.graphPanel.region.setRegion([left, right])
        self.graphPanel.region.sigRegionChangeFinished.connect(self.range_to_dialog)
        self.controlPanel.settingPanel.spinBox_pixel_range_left.setValue(left)
        self.controlPanel.settingPanel.spinBox_pixel_range_right.setValue(right)


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
            lbl_slice_count = QtWidgets.QLabel("slice count")
            lbl_center = QtWidgets.QLabel("center")
            self.spinBox_irange1 = QtWidgets.QSpinBox()
            self.spinBox_irange2 = QtWidgets.QSpinBox()
            self.spinBox_slice_count = QtWidgets.QSpinBox()
            self.spinBox_center_x = QtWidgets.QSpinBox()
            self.spinBox_center_y = QtWidgets.QSpinBox()
            self.chkBox_show_centerLine = QtWidgets.QCheckBox("Show center line")
            self.chkBox_show_beam_stopper_mask = QtWidgets.QCheckBox("Show beam stopper mask")
            self.spinBox_irange1.setMinimum(1)
            self.spinBox_irange2.setMinimum(1)
            self.spinBox_slice_count.setMinimum(1)
            self.spinBox_center_x.setMinimum(1)
            self.spinBox_center_y.setMinimum(1)
            self.spinBox_irange1.setMaximum(255)
            self.spinBox_irange2.setMaximum(255)
            self.spinBox_slice_count.setMaximum(255)
            self.spinBox_irange1.setValue(util.settings["intensity_range_1"])
            self.spinBox_irange2.setValue(util.settings["intensity_range_2"])
            self.spinBox_slice_count.setValue(util.settings["slice_count"])
            self.chkBox_show_centerLine.setChecked(util.settings["show_center_line"])
            self.chkBox_show_beam_stopper_mask.setChecked(util.settings["show_beam_stopper_mask"])
            # self.spinBox_irange1.setFixedHeight(ControlPanel.text_fixed_height)
            # self.spinBox_irange2.setFixedHeight(ControlPanel.text_fixed_height)
            # self.spinBox_slice_count.setFixedHeight(ControlPanel.text_fixed_height)
            # self.spinBox_irange1.setFixedWidth(ControlPanel.text_fixed_width)
            # self.spinBox_irange2.setFixedWidth(ControlPanel.text_fixed_width)
            # self.spinBox_slice_count.setFixedWidth(ControlPanel.text_fixed_width)

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
            layout.addWidget(lbl_slice_count, 2, 0, 1, 2)
            layout.addWidget(self.spinBox_irange1, 1, 2)
            layout.addWidget(self.spinBox_irange2, 1, 3)
            layout.addWidget(self.spinBox_slice_count, 2, 2)
            layout.addWidget(lbl_center,3,0,1,2)
            layout.addWidget(self.spinBox_center_x,3,2)
            layout.addWidget(self.spinBox_center_y, 3,3)
            layout.addWidget(self.chkBox_show_centerLine,4,0,1,4)
            layout.addWidget(self.chkBox_show_beam_stopper_mask,5,0,1,4)


            lbl_pixel_range = QtWidgets.QLabel("pixel Range")
            self.spinBox_pixel_range_left = QtWidgets.QSpinBox()
            self.spinBox_pixel_range_right = QtWidgets.QSpinBox()

            layout.addWidget(lbl_pixel_range, 6, 0, 1, 2)
            layout.addWidget(self.spinBox_pixel_range_left, 6, 2)
            layout.addWidget(self.spinBox_pixel_range_right, 6, 3)

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


import cv2

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
        self._current_data = img
        if len(img.shape) == 2:
            self.imageView.setImage(self._current_data.transpose(1,0))
        if len(img.shape) == 3:
            self.imageView.setImage(self._current_data.transpose(1,0,2))
    def get_img(self):
        return self._current_data


class GraphPanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.imageView = pg.ImageView()
        self.plot_azav = pg.PlotWidget(title='azimuthal average')
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.plot_azav)
        self.setLayout(self.layout)
        self.setMinimumHeight(200)
        self.region = pg.LinearRegionItem([0, 100])
        self.plot_azav.addItem(self.region)

        self.legend = self.plot_azav.addLegend(offset=(-30,30))

        self.button_grp_widget = QtWidgets.QWidget()
        self.button_grp_widget.layout = QtWidgets.QVBoxLayout()
        self.button_grp_widget.setLayout(self.button_grp_widget.layout)
        self.button_start = QtWidgets.QPushButton("start")
        self.button_all = QtWidgets.QPushButton("all")
        self.button_end = QtWidgets.QPushButton("end")
        self.button_grp_widget.layout.addWidget(self.button_start)
        self.button_grp_widget.layout.addWidget(self.button_all)
        self.button_grp_widget.layout.addWidget(self.button_end)

        self.layout.addWidget(self.button_grp_widget)

        self.azav_curr_dat = None
        self.azav_prev_dat = None
        self.plot_azav_curr = self.plot_azav.plot( pen=pg.mkPen(255, 0, 0, width=2), name='current')
        self.plot_azav_prev = self.plot_azav.plot( pen=pg.mkPen(0, 255, 0, width=2), name='previous')


        # self.setMaximumHeight(300)

    def update_graph(self, dat):
        # self.plotWindow.layout.setSpacing(0)
        # self.plotWindow.layout.setContentsMargins(0,0,0,0)
        # self.plot_azav = pg.PlotWidget(title='azimuthal average')
        # self.plotWindow.layout.addWidget(self.plot_azav)
        # self.plotWindow.setLayout(self.plotWindow.layout)

        if self.azav_curr_dat is None:
            self.azav_curr_dat = dat
            self.plot_azav_curr.setData(self.azav_curr_dat)
        else :
            self.azav_prev_dat = self.azav_curr_dat
            self.azav_curr_dat = dat
            self.plot_azav_curr.setData(self.azav_curr_dat)
            self.plot_azav_prev.setData(self.azav_prev_dat)

        # self.plotWindow.resize(1000,350)




if __name__ == '__main__':
    app = DataViewer(sys.argv)
    sys.exit(app.qtapp.exec_())