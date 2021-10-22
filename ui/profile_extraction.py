from PyQt5 import QtCore, QtWidgets, QtGui
import os, sys
import pyqtgraph as pg
import file
import numpy as np
import util
import time
from datacube import DataCube
from typing import List
from ui.pdfanalysis import PdfAnalysis
from calculate import pdf_calculator, image_process, q_range_selector
from PyQt5.QtWidgets import QMessageBox
from ui import ui_util
pg.setConfigOptions(antialias=True)
import definitions
import cv2


class ProfileExtraction(QtWidgets.QWidget):
    def __init__(self, Dataviewer):
        QtWidgets.QWidget.__init__(self)
        self.Dataviewer = Dataviewer
        self.init_ui()
        self.default_setting = util.DefaultSetting()
        self.isShowCenter = True
        self.flag_range_update = False
        self.load_default()
        self.sig_binding()
        # for text
        # self.menu_open_azavg_only(np.loadtxt("/mnt/experiment/TEM diffraction/201126 (test)/sample38_TiTa_annealed/Analysis ePDFpy/Camera 230 mm Ceta 20201126 1649_40s_20f_area01.azavg.txt"))
        # self.menu_open_azavg_only(np.loadtxt(r"V:\experiment\TEM diffraction\201126 (test)\sample38_TiTa_annealed\Analysis ePDFpy\Camera 230 mm Ceta 20201126 1649_40s_20f_area01.azavg.txt"))
        ##

    def init_ui(self):
        self.control_panel = ControlPanel()
        self.img_panel = ImgPanel()
        self.profile_graph_panel = IntensityProfilePanel()
        self.std_graph_panel = StdGraphPanel()

        self.upper_left = self.control_panel
        self.bottom_left = self.img_panel
        self.upper_right = self.profile_graph_panel
        self.bottom_right = self.std_graph_panel

        self.splitter_left_vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter_left_vertical.addWidget(self.upper_left)
        self.splitter_left_vertical.addWidget(self.bottom_left)
        self.splitter_left_vertical.setStretchFactor(1, 1)

        self.splitter_right_vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter_right_vertical.addWidget(self.upper_right)
        self.splitter_right_vertical.addWidget(self.bottom_right)

        self.left = self.splitter_left_vertical
        self.right = self.splitter_right_vertical

        self.splitter_horizontal = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter_horizontal.addWidget(self.left)
        self.splitter_horizontal.addWidget(self.right)
        self.splitter_horizontal.setStretchFactor(0, 10)
        self.splitter_horizontal.setStretchFactor(1, 7)

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(3, 3, 3, 3)
        self.layout.addWidget(self.splitter_horizontal)

        self.setLayout(self.layout)

    def load_default(self):
        if util.default_setting.intensity_range_1 is not None:
            self.control_panel.settingPanel.spinBox_irange1.setValue(util.default_setting.intensity_range_1)
        if util.default_setting.intensity_range_2 is not None:
            self.control_panel.settingPanel.spinBox_irange2.setValue(util.default_setting.intensity_range_2)
        if util.default_setting.slice_count is not None:
            self.control_panel.settingPanel.spinBox_slice_count.setValue(util.default_setting.slice_count)
        if util.default_setting.show_center_line is not None:
            self.control_panel.settingPanel.chkBox_show_centerLine.setChecked(util.default_setting.show_center_line)
        if util.default_setting.show_beam_stopper_mask is not None:
            self.control_panel.settingPanel.chkBox_show_beam_stopper_mask.setChecked(
                util.default_setting.show_beam_stopper_mask)

    def sig_binding(self):
        # self.controlPanel.openFilePanel.open_img_file.triggered.connect(self.menu_open_image_file)
        # self.controlPanel.openFilePanel.open_img_folder.triggered.connect(self.menu_open_image_stack)
        # self.controlPanel.openFilePanel.open_preset.triggered.connect(self.menu_load_preset)
        # self.controlPanel.openFilePanel.save_preset.triggered.connect(self.menu_save_preset)
        # self.controlPanel.openFilePanel.open_presets.triggered.connect(self.menu_open_preset_stack)
        # self.controlPanel.openFilePanel.save_presets.triggered.connect(self.menu_save_presets)
        # self.controlPanel.openFilePanel.open_azavg_only.triggered.connect(self.menu_open_azavg_only)
        # self.controlPanel.openFilePanel.save_azavg_only.triggered.connect(self.menu_save_azavg_only)
        # self.controlPanel.openFilePanel.combo_dataQuality.currentIndexChanged.connect(self.set_data_quality)

        self.control_panel.operationPanel.btn_find_center.clicked.connect(lambda: (self.find_center(),self.update_img()))
        self.control_panel.operationPanel.btn_get_azimuthal_avg.clicked.connect(self.get_azimuthal_value)
        self.control_panel.settingPanel.spinBox_center_x.valueChanged.connect(self.spinbox_changed_event)
        self.control_panel.settingPanel.spinBox_center_y.valueChanged.connect(self.spinbox_changed_event)
        self.control_panel.operationPanel.btn_calculate_all_azimuthal.clicked.connect(self.calculate_all_azimuthal)
        self.control_panel.settingPanel.chkBox_show_centerLine.stateChanged.connect(self.update_img)
        self.control_panel.settingPanel.chkBox_show_beam_stopper_mask.stateChanged.connect(self.update_img)

    def spinbox_changed_event(self):
        x = self.control_panel.settingPanel.spinBox_center_x.value()
        y = self.control_panel.settingPanel.spinBox_center_y.value()
        self.dc.center = (x, y)
        self.update_img()

    def calculate_all_azimuthal(self):
        tic = time.time()
        for i in range(len(self.Dataviewer.dcs)):
            print("processing azimuthal values", self.dc.mrc_file_path)
            self.Dataviewer.load_dc(i)
            if self.Dataviewer.dcs[i].img is not None and self.Dataviewer.dcs[i].center[0] is None:
                self.find_center()
                self.update_img()
            if self.Dataviewer.dcs[i].azavg is None:
                self.get_azimuthal_value()
            toc = time.time()
            print("({}/{}) Done, Time:{}".format(i+1,len(self.Dataviewer.dcs),toc-tic))
        toc = time.time()
        print("Calculation is done, {}".format(toc-tic))
            # self.datacubes[i].save_azimuthal_data()
            # self.save_current_azimuthal()
        #     self.controlPanel.operationPanel.progress_bar.setValue((i+1)/len(self.datacubes))
        # self.controlPanel.operationPanel.progress_bar.setValue(0)

    def get_azimuthal_value(self):

        # calculate azavg
        i1 = self.control_panel.settingPanel.spinBox_irange1.value()
        i2 = self.control_panel.settingPanel.spinBox_irange2.value()
        intensity_range = (i1, i2)
        slice_count = int(self.control_panel.settingPanel.spinBox_slice_count.value())
        self.dc.calculate_azimuthal_average(intensity_range, slice_count)

        # update ui
        self.update_azavg_graph()
        self.update_std_graph()
        self.Dataviewer.PDF_analyser.update_initial_iq()
        self.Dataviewer.PDF_analyser.update_initial_iq_graph()
        self.update_center_spinBox()
        self.update_img()

    def update_azavg_graph(self):
        if self.dc.azavg is None:
            return
        self.profile_graph_panel.update_graph(self.dc.azavg)

    def update_std_graph(self):
        if self.dc.azvar is None:
            return
        self.std_graph_panel.update_graph(self.dc.azvar)


    def find_center(self):
        i1 = self.control_panel.settingPanel.spinBox_irange1.value()
        i2 = self.control_panel.settingPanel.spinBox_irange2.value()
        intensity_range = (i1, i2)
        slice_count = int(self.control_panel.settingPanel.spinBox_slice_count.value())
        self.dc.calculate_center(intensity_range, slice_count)
        self.update_center_spinBox()
        # you must use self.draw_center() after find_center
        return self.dc.center

    def update_dc(self, dc):
        self.dc = dc

        # update number
        # self.imgPanel.lbl_current_num.setText(str(self.current_page + 1) + "/" + str(len(self.dcs)))

        # update quality number
        # self.reload_auto_data_quality()
        # if self.dc.data_quality_idx is not None:
        #     self.controlPanel.openFilePanel.combo_dataQuality.setCurrentIndex(self.dc.data_quality_idx)
        # else:
        #     self.controlPanel.openFilePanel.combo_dataQuality.setCurrentIndex(0)

        # update img
        self.update_img()

        # update graph
        self.update_azavg_graph()
        self.update_std_graph()

        # update spinbox and settings
        self.update_center_spinBox()

        # windows title : file name
        # if dc.load_file_path is not None:
        #     self.setWindowTitle(self.dc.load_file_path)
        #     fn = os.path.split(self.dc.load_file_path)[1]
        #     max_size = 25
        #     if len(fn) > max_size:
        #         self.controlPanel.openFilePanel.lbl_file_name.setText("..."+fn[-max_size:])
        #     else:
        #         self.controlPanel.openFilePanel.lbl_file_name.setText(fn)

    def update_center_spinBox(self):
        if not self.dc.img is None:
            self.control_panel.settingPanel.spinBox_center_x.setMaximum(self.dc.img.shape[0])  # todo : confusing x,y
            self.control_panel.settingPanel.spinBox_center_y.setMaximum(self.dc.img.shape[1])

        if not self.dc.center[0] is None:
            ui_util.update_value(self.control_panel.settingPanel.spinBox_center_x, self.dc.center[0])
            ui_util.update_value(self.control_panel.settingPanel.spinBox_center_y, self.dc.center[1])

    def update_img(self):
        if self.dc.img is None:
            self.img_panel.clear_img()
            return
        img = self.dc.img.copy()
        if self.control_panel.settingPanel.chkBox_show_beam_stopper_mask.isChecked():
            img = cv2.bitwise_and(img, img, mask=np.bitwise_not(image_process.mask))
        if self.dc.center[0] is not None and self.control_panel.settingPanel.chkBox_show_centerLine.isChecked():
            img = image_process.draw_center_line(img, self.dc.center)
        self.img_panel.update_img(img)

    def set_data_quality(self):
        # set auto(data quality) combolist and save to datacube
        txt_auto_quality = self.reload_auto_data_quality()
        combobox_idx = self.control_panel.openFilePanel.combo_dataQuality.currentIndex()
        self.dc.data_quality_idx = combobox_idx
        if combobox_idx == 0:
            # None
            self.dc.data_quality = None
        elif combobox_idx == 1:
            # Auto
            self.dc.data_quality = txt_auto_quality
        else:
            # Manual Selection of L1, L2, L3, L4
            self.dc.data_quality = self.control_panel.openFilePanel.combo_dataQuality.currentText()

    def reload_auto_data_quality(self):
        if self.dc.pixel_end_n is None:
            return
        right = self.dc.pixel_end_n
        txt_auto_quality = util.get_data_quality(right)
        self.control_panel.openFilePanel.combo_dataQuality.setItemText(1, "Auto({})".format(txt_auto_quality))
        return txt_auto_quality


class ControlPanel(QtWidgets.QWidget):
    # text_fixed_height = 25
    # text_fixed_width = 70

    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        # self.openFilePanel = self.OpenFilePanel("OpenFile", mainWindow)
        self.settingPanel = self.SettingPanel("Center finding setting")
        self.operationPanel = self.OperationPanel("Operation")

        layout = QtWidgets.QHBoxLayout()
        # layout.addWidget(self.openFilePanel)
        layout.addWidget(self.settingPanel)
        layout.addWidget(self.operationPanel)
        self.setLayout(layout)

    # class OpenFilePanel(QtWidgets.QGroupBox):
    #     def __init__(self,arg,mainWindow: QtWidgets.QMainWindow):
    #         QtWidgets.QGroupBox.__init__(self,arg)
    #         layout = QtWidgets.QGridLayout()
    #
    #         radio_grp = QtWidgets.QGroupBox()
    #         self.radio_folder = QtWidgets.QRadioButton("Folder")
    #         self.radio_file = QtWidgets.QRadioButton("File")
    #         radio_grp_layout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight)
    #         radio_grp.setLayout(radio_grp_layout)
    #         radio_grp_layout.addWidget(self.radio_folder)
    #         radio_grp_layout.addWidget(self.radio_file)
    #         self.radio_file.setChecked(True)
    #
    #         self.lbl_path = QtWidgets.QLabel("/")
    #         self.btn_path = QtWidgets.QPushButton("open")
    #         # self.lbl_path.setFixedHeight(ControlPanel.text_fixed_height)
    #         self.lbl_path.setMaximumWidth(300)
    #
    #
    #         layout.addWidget(radio_grp, 0, 0)
    #         layout.addWidget(self.btn_path, 0, 1)
    #         layout.addWidget(self.lbl_path, 1, 0, 1, 2)
    #         self.setLayout(layout)

    class OpenFilePanel(QtWidgets.QGroupBox):
        def __init__(self, arg, mainWindow: QtWidgets.QMainWindow):
            QtWidgets.QGroupBox.__init__(self)
            self.setTitle(arg)
            layout = QtWidgets.QGridLayout()
            layout = QtWidgets.QHBoxLayout()
            self.menu_file = self.create_menu(mainWindow)
            self.lbl_file_name = QtWidgets.QLabel("...")
            layout.addWidget(self.menu_file)
            layout.addWidget(self.lbl_file_name)
            layout.addWidget(self.menu_file, 0, 0, 1, 1)
            layout.addWidget(self.lbl_file_name, 0, 1, 1, 2)

            self.lbl_data_quality = QtWidgets.QLabel("Data Quality")
            self.combo_dataQuality = QtWidgets.QComboBox()
            quality_list = ["None", "Auto"]
            quality_list.extend(util.df_data_quality['label'].to_list())  # [None,Auto,L1,L2,L3,L4]
            self.combo_dataQuality.addItems(quality_list)
            layout.addWidget(self.combo_dataQuality, 1, 1, 1, 1)
            layout.addWidget(self.lbl_data_quality, 1, 0, 1, 1)

            self.setLayout(layout)

        def create_menu(self, mainWindow: QtWidgets.QMainWindow):
            menubar = mainWindow.menuBar()
            menubar.setNativeMenuBar(False)
            self.open_img_file = QtWidgets.QAction("Open &image file", self)
            self.open_img_stack = QtWidgets.QAction("Open &image stack", self)
            self.open_preset = QtWidgets.QAction("Open preset &file", self)
            self.save_preset = QtWidgets.QAction("Save preset &file", self)
            self.open_preset_stack = QtWidgets.QAction("Open preset &stack", self)
            self.save_preset_stack = QtWidgets.QAction("Save preset &stack", self)
            self.open_azavg_only = QtWidgets.QAction("Open &azavg only", self)
            self.save_azavg_only = QtWidgets.QAction("Save &azavg only", self)

            open_menu = menubar.addMenu("     &Open     ")
            open_menu.addAction(self.open_img_file)
            open_menu.addAction(self.open_img_stack)
            open_menu.addSeparator()
            open_menu.addAction(self.open_preset)
            open_menu.addAction(self.open_preset_stack)
            open_menu.addSeparator()
            open_menu.addAction(self.open_azavg_only)

            open_menu = menubar.addMenu("     &Save     ")
            open_menu.addAction(self.save_preset)
            open_menu.addAction(self.save_preset_stack)
            open_menu.addAction(self.save_azavg_only)

            menubar.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            return menubar

    class SettingPanel(QtWidgets.QGroupBox):
        def __init__(self, arg):
            QtWidgets.QGroupBox.__init__(self, arg)
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
            layout.addWidget(lbl_center, 3, 0, 1, 2)
            layout.addWidget(self.spinBox_center_x, 3, 2)
            layout.addWidget(self.spinBox_center_y, 3, 3)
            layout.addWidget(self.chkBox_show_centerLine, 4, 0, 1, 4)
            layout.addWidget(self.chkBox_show_beam_stopper_mask, 5, 0, 1, 4)

            # lbl_pixel_range = QtWidgets.QLabel("pixel Range")
            # self.spinBox_pixel_range_left = QtWidgets.QSpinBox()
            # self.spinBox_pixel_range_right = QtWidgets.QSpinBox()

            # layout.addWidget(lbl_pixel_range, 6, 0, 1, 2)
            # layout.addWidget(self.spinBox_pixel_range_left, 6, 2)
            # layout.addWidget(self.spinBox_pixel_range_right, 6, 3)

            self.setLayout(layout)

    class OperationPanel(QtWidgets.QGroupBox):
        def __init__(self, arg):
            QtWidgets.QGroupBox.__init__(self, arg)
            layout = QtWidgets.QGridLayout()
            self.btn_find_center = QtWidgets.QPushButton("Find center")
            self.btn_get_azimuthal_avg = QtWidgets.QPushButton("Get azimuthal data")
            self.btn_calculate_all_azimuthal = QtWidgets.QPushButton("Calculate all data")
            self.progress_bar = QtWidgets.QProgressBar()
            self.progress_bar.setValue(0)

            layout.addWidget(self.btn_find_center, 0, 0)
            layout.addWidget(self.btn_get_azimuthal_avg, 0, 1)
            # layout.addWidget(self.btn_save_current_azimuthal, 1, 0,1,2)
            layout.addWidget(self.btn_calculate_all_azimuthal, 2, 0, 1, 2)
            layout.addWidget(self.progress_bar, 3, 0, 1, 2)

            self.setLayout(layout)



class ImgPanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.imageView = pg.ImageView()
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.imageView, 0, 0, 1, 9)
        self.setLayout(layout)
        self._current_data = None
        self.cmap = pg.ColorMap(np.linspace(0, 1, len(image_process.colorcube)), color=image_process.colorcube)

    def update_img(self, img):
        self.imageView.setColorMap(self.cmap)
        self._current_data = img
        if len(img.shape) == 2:
            self.imageView.setImage(self._current_data.transpose(1, 0))
        if len(img.shape) == 3:
            self.imageView.setImage(self._current_data.transpose(1, 0, 2))

    def clear_img(self):
        self.imageView.clear()

    def get_img(self):
        return self._current_data


class IntensityProfilePanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.plot_widget = ui_util.IntensityPlotWidget(title='Intensity Profile')
        self.plot_widget.setYScaling(False)
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.plot_widget)
        self.setLayout(self.layout)

        self.curr_dat = None
        self.prev_dat = None
        self.plot_widget.addLegend(offset=(-30, 30))
        self.plot_prev = self.plot_widget.plot(pen=pg.mkPen(100, 100, 100, width=2), name='previous')
        self.plot_curr = self.plot_widget.plot(pen=pg.mkPen(255, 0, 0, width=2), name='current')


    def update_graph(self, dat):
        if self.curr_dat is None:
            self.curr_dat = dat
            self.plot_curr.setData(self.curr_dat)
        else:
            self.prev_dat = self.curr_dat
            self.curr_dat = dat
            self.plot_curr.setData(self.curr_dat)
            self.plot_prev.setData(self.prev_dat)

class StdGraphPanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.plot_widget = ui_util.IntensityPlotWidget(title='Normalized std')
        self.plot_widget.setYScaling(False)
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.plot_widget)
        self.setLayout(self.layout)

        self.curr_dat = None
        self.prev_dat = None
        self.plot_widget.addLegend(offset=(-30, 30))
        self.plot_prev = self.plot_widget.plot(pen=pg.mkPen(100, 100, 100, width=2), name='previous')
        self.plot_curr = self.plot_widget.plot(pen=pg.mkPen(255, 0, 0, width=2), name='current')


    def update_graph(self, dat):
        if self.curr_dat is None:
            self.curr_dat = dat
            self.plot_curr.setData(self.curr_dat)
        else:
            self.prev_dat = self.curr_dat
            self.curr_dat = dat
            self.plot_curr.setData(self.curr_dat)
            self.plot_prev.setData(self.prev_dat)
