from PyQt5 import QtCore, QtWidgets, QtGui
import os, sys
import pyqtgraph as pg
import file
import numpy as np
import util
from datacube import DataCube
from typing import List
from ui.pdf_analyse import pdf_analyse
from calculate import pdf_calculator, image_process, q_range_selector
from PyQt5.QtWidgets import QMessageBox
from ui import ui_util
pg.setConfigOptions(antialias=True)

class DataViewer(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.init_ui()
        self.isShowCenter=True
        self.flag_range_update = False
        self.load_default()
        self.sig_binding()
        self.dcs: List[DataCube] = []
        self.resize(1000,600)
        # for text
        # self.menu_open_azavg_only(np.loadtxt("/mnt/experiment/TEM diffraction/201126 (test)/sample38_TiTa_annealed/Analysis ePDFpy/Camera 230 mm Ceta 20201126 1649_40s_20f_area01.azavg.txt"))
        # self.menu_open_azavg_only(np.loadtxt(r"V:\experiment\TEM diffraction\201126 (test)\sample38_TiTa_annealed\Analysis ePDFpy\Camera 230 mm Ceta 20201126 1649_40s_20f_area01.azavg.txt"))
        ##

    def init_ui(self):
        QtWidgets.QWidget.__init__(self)
        self.plotWindow = None
        self.eRDF_analyser = None
        self.current_page = 0
        self.upper = QtWidgets.QFrame(self)
        self.lower = QtWidgets.QFrame(self)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.controlPanel = ControlPanel(self)
        self.controlPanel.setMaximumWidth(300)
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

        self.setWindowTitle("Main window")
        self.default_setting = util.DefaultSetting()

        centralWidget = QtWidgets.QWidget()
        centralWidget.setLayout(self.layout)
        self.setCentralWidget(centralWidget)


    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        util.default_setting.intensity_range_1 = self.controlPanel.settingPanel.spinBox_irange1.value()
        util.default_setting.intensity_range_2 = self.controlPanel.settingPanel.spinBox_irange2.value()
        util.default_setting.slice_count = self.controlPanel.settingPanel.spinBox_slice_count.value()
        util.default_setting.show_center_line = self.controlPanel.settingPanel.chkBox_show_centerLine.isChecked()
        util.default_setting.save_settings()
        super().closeEvent(a0)

    def load_default(self):
        if util.default_setting.intensity_range_1 is not None:
            self.controlPanel.settingPanel.spinBox_irange1.setValue(util.default_setting.intensity_range_1)
        if util.default_setting.intensity_range_2 is not None:
            self.controlPanel.settingPanel.spinBox_irange2.setValue(util.default_setting.intensity_range_2)
        if util.default_setting.slice_count is not None:
            self.controlPanel.settingPanel.spinBox_slice_count.setValue(util.default_setting.slice_count)
        if util.default_setting.show_center_line is not None:
            self.controlPanel.settingPanel.chkBox_show_centerLine.setChecked(util.default_setting.show_center_line)
        if util.default_setting.show_beam_stopper_mask is not None:
            self.controlPanel.settingPanel.chkBox_show_beam_stopper_mask.setChecked(util.default_setting.show_beam_stopper_mask)

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
            self.btn_page_left_clicked()
        if e.key() == QtCore.Qt.Key.Key_PageDown:
            self.btn_page_right_clicked()

    def sig_binding(self):
        self.controlPanel.openFilePanel.open_img_file.triggered.connect(self.menu_open_image_file)
        self.controlPanel.openFilePanel.open_img_folder.triggered.connect(self.menu_open_image_folder)
        self.controlPanel.openFilePanel.open_preset.triggered.connect(self.menu_load_preset)
        self.controlPanel.openFilePanel.save_preset.triggered.connect(self.menu_save_preset)
        self.controlPanel.openFilePanel.open_presets.triggered.connect(self.menu_open_preset_folder)
        self.controlPanel.openFilePanel.save_presets.triggered.connect(self.menu_save_presets)
        self.controlPanel.openFilePanel.open_azavg_only.triggered.connect(self.menu_open_azavg_only)
        self.controlPanel.openFilePanel.save_azavg_only.triggered.connect(self.menu_save_azavg_only)

        self.controlPanel.operationPanel.btn_find_center.clicked.connect(lambda: (self.find_center(),self.update_img()))
        self.imgPanel.btn_left.clicked.connect(self.btn_page_left_clicked)
        self.imgPanel.btn_right.clicked.connect(self.btn_page_right_clicked)
        self.controlPanel.operationPanel.btn_get_azimuthal_avg.clicked.connect(self.get_azimuthal_value)
        self.controlPanel.settingPanel.spinBox_center_x.valueChanged.connect(self.spinbox_changed_event)
        self.controlPanel.settingPanel.spinBox_center_y.valueChanged.connect(self.spinbox_changed_event)
        # self.controlPanel.operationPanel.btn_save_current_azimuthal.clicked.connect(self.save_current_azimuthal)
        self.controlPanel.operationPanel.btn_calculate_all_azimuthal.clicked.connect(self.save_all_azimuthal)
        self.controlPanel.settingPanel.chkBox_show_centerLine.stateChanged.connect(self.update_img)
        self.controlPanel.settingPanel.chkBox_show_beam_stopper_mask.stateChanged.connect(self.update_img)
        self.graphPanel.spinBox_pixel_range_left.valueChanged.connect(self.dialog_to_range)
        self.graphPanel.spinBox_pixel_range_right.valueChanged.connect(self.dialog_to_range)
        self.graphPanel.region.sigRegionChangeFinished.connect(self.range_to_dialog)
        self.graphPanel.button_start.clicked.connect(self.btn_range_start_clicked)
        self.graphPanel.button_all.clicked.connect(self.btn_range_all_clicked)
        self.graphPanel.button_end.clicked.connect(self.btn_range_end_clicked)
        self.graphPanel.button_select.clicked.connect(self.btn_select_clicked)
        self.controlPanel.operationPanel.btn_open_epdf_analyser.clicked.connect(self.btn_show_erdf_analyser)

    def btn_select_clicked(self):
        azavg = self.dcs[self.current_page].azavg
        if self.dcs[self.current_page].azavg is None:
            return
        first_peak_idx, second_peak_idx = q_range_selector.find_multiple_peaks(self.dcs[self.current_page].azavg)
        self.graphPanel.plot_azav.create_circle([first_peak_idx,azavg[first_peak_idx]],[second_peak_idx,azavg[second_peak_idx]])

        #
        self.upper.hide()

        l = q_range_selector.find_first_nonzero_idx(self.dcs[self.current_page].azavg)
        r = l + int((len(self.dcs[self.current_page].azavg) - l) / 4)
        self.graphPanel.plot_azav.setXRange(l, r, padding=0.1)
        
        self.graphPanel.plot_azav.select_mode = True
        self.graphPanel.plot_azav.select_event = self.azav_select_event
        #
        # self.upper.show()

    def azav_select_event(self):
        self.upper.show()
        self.graphPanel.plot_azav.select_mode = False
        self.graphPanel.plot_azav.first_dev_plot.clear()
        self.graphPanel.plot_azav.first_dev_plot = None
        self.graphPanel.plot_azav.second_dev_plot.clear()
        self.graphPanel.plot_azav.second_dev_plot = None

    def spinbox_changed_event(self):
        x = self.controlPanel.settingPanel.spinBox_center_x.value()
        y = self.controlPanel.settingPanel.spinBox_center_y.value()
        self.dcs[self.current_page].center = (x, y)
        self.update_img()

    def btn_show_erdf_analyser(self):
        if len(self.dcs) == 0:
            self.eRDF_analyser = pdf_analyse(DataCube())
            return
        self.eRDF_analyser = pdf_analyse(self.dcs[self.current_page])

    def btn_range_start_clicked(self):
        # left = q_range_selector.find_first_nonzero_idx(self.dcs[self.current_page].azavg)
        left = self.graphPanel.spinBox_pixel_range_left.value()
        right = self.graphPanel.spinBox_pixel_range_right.value()

        l = left
        r = left + int((right - left) / 4)
        # r = left + int((len(self.dcs[self.current_page].azavg) - left) / 4)
        # print("left {}, right {}".format(l, r))
        # mx = np.max(self.dcs[self.current_page].azavg[l:r])
        # mn = np.min(self.dcs[self.current_page].azavg[l:r])
        self.graphPanel.plot_azav.setXRange(l, r, padding=0.1)
        # self.graphPanel.plot_azav.setYRange(mn, mx, padding=0.1)
        # print(self.graphPanel.plot_azav.viewRange())

    def btn_range_all_clicked(self):
        self.graphPanel.plot_azav.autoRange()

    def btn_range_end_clicked(self):
        left = self.graphPanel.spinBox_pixel_range_left.value()
        right = self.graphPanel.spinBox_pixel_range_right.value()
        l = right-int((right - left) / 4)
        r = right
        # mx = np.max(self.dcs[self.current_page].azavg[l:r])
        # mn = np.min(self.dcs[self.current_page].azavg[l:r])
        self.graphPanel.plot_azav.setXRange(l, r, padding=0.1)
        # self.graphPanel.plot_azav.setYRange(mn, mx, padding=0.1)

    def save_current_azimuthal(self):
        self.dcs[self.current_page]\
            .save_azimuthal_data(intensity_start=self.controlPanel.settingPanel.spinBox_irange1.value(),
                                 intensity_end=self.controlPanel.settingPanel.spinBox_irange2.value(),
                                 intensity_slice=self.controlPanel.settingPanel.spinBox_slice_count.value(),
                                 imgPanel=self.imgPanel,
                                 draw_center_line=self.controlPanel.settingPanel.chkBox_show_centerLine.isChecked(),
                                 masking=self.controlPanel.settingPanel.chkBox_show_beam_stopper_mask.isChecked()
                                 )

    def save_all_azimuthal(self):
        for i in range(len(self.dcs)):
            print("processing azimuthal values", self.dcs[self.current_page].mrc_file_path)
            self.update_ui_dc(i)
            if self.dcs[i].img is not None and self.dcs[i].center[0] is None:
                self.find_center()
                self.update_img()
            if self.dcs[i].azavg is None:
                self.get_azimuthal_value()
            # self.datacubes[i].save_azimuthal_data()
            # self.save_current_azimuthal()
        #     self.controlPanel.operationPanel.progress_bar.setValue((i+1)/len(self.datacubes))
        # self.controlPanel.operationPanel.progress_bar.setValue(0)

    def get_azimuthal_value(self):
        i1 = self.controlPanel.settingPanel.spinBox_irange1.value()
        i2 = self.controlPanel.settingPanel.spinBox_irange2.value()
        intensity_range = (i1,i2)
        slice_count = int(self.controlPanel.settingPanel.spinBox_slice_count.value())
        self.dcs[self.current_page].calculate_azimuthal_average(intensity_range,slice_count)
        self.update_azavg_graph()
        self.update_center_spinBox()
        self.update_img()

    def update_azavg_graph(self):
        if self.dcs[self.current_page].azavg is None:
            return
        self.graphPanel.update_graph(self.dcs[self.current_page].azavg)
        self.graphPanel.spinBox_pixel_range_right.setMaximum(len(self.dcs[self.current_page].azavg))
        self.graphPanel.spinBox_pixel_range_left.setMaximum(len(self.dcs[self.current_page].azavg))

        if self.dcs[self.current_page].pixel_start_n is None:
            left = q_range_selector.find_first_peak(self.dcs[self.current_page].azavg)
            # left = 0
            # for i in range(len(self.datacubes[self.current_page].azavg)):
            #     if int(self.datacubes[self.current_page].azavg[i]) != 0 :
            #         left = i
            #         break
            self.graphPanel.region.setRegion([left, len(self.dcs[self.current_page].azavg)-1])
        else:
            self.graphPanel.region.setRegion([self.dcs[self.current_page].pixel_start_n,self.dcs[self.current_page].pixel_end_n])

    def find_center(self):
        i1 = self.controlPanel.settingPanel.spinBox_irange1.value()
        i2 = self.controlPanel.settingPanel.spinBox_irange2.value()
        intensity_range = (i1,i2)
        slice_count = int(self.controlPanel.settingPanel.spinBox_slice_count.value())
        self.dcs[self.current_page].calculate_center(intensity_range, slice_count)
        self.update_center_spinBox()
        # you must use self.draw_center() after find_center
        return self.dcs[self.current_page].center

    def menu_open_image_file(self):
        load_paths = []
        path,_ = QtWidgets.QFileDialog.getOpenFileNames(self,'open')
        if len(path) == 0:
            return
        load_paths.extend(path)
        self.dcs.clear()
        self.dcs.extend([DataCube(path,'image') for path in load_paths])
        self.update_ui_dc(0)

    def menu_open_image_folder(self):
        load_paths = []
        path = QtWidgets.QFileDialog.getExistingDirectory(self,'open')
        if len(path) == 0:
            return
        load_paths.extend(file.get_file_list_from_path(path,'.mrc'))
        if len(load_paths) == 0:
            QMessageBox.about(self,"No file found","No file found")
            return
        self.dcs.clear()
        self.dcs.extend([DataCube(path,'image') for path in load_paths])
        self.current_page = 0
        self.update_ui_dc(0)

    def menu_open_preset_folder(self):
        load_paths = []
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'open')
        if len(path) == 0:
            return
        load_paths.extend(file.get_file_list_from_path(path, file.preset_ext))
        if len(load_paths) == 0:
            QMessageBox.about(self,"No file found","No file found")
            return
        self.dcs.clear()
        self.dcs.extend([file.load_preset(path) for path in load_paths])
        self.current_page = 0
        self.update_ui_dc(0)

    def menu_open_azavg_only(self, azavg=None):  # azavg arguments is for averaging_multiple_gr.py
        if azavg is None or azavg is False:
            fp, _ = QtWidgets.QFileDialog.getOpenFileName()
            if fp is '':
                return
            dc = DataCube(file_path=fp,file_type='azavg')
            self.dcs.clear()
            self.dcs.append(dc)
        else:
            self.dcs.clear()
            self.dcs.append(DataCube())
            self.dcs[0].azavg = azavg
        self.update_ui_dc(0)

    def menu_load_preset(self):
        dc = file.load_preset()
        if not dc:
            return
        self.dcs.clear()
        self.dcs.append(dc)
        self.update_ui_dc(0)

    def menu_save_preset(self):
        file.save_preset_default(self.dcs[self.current_page], self.imgPanel)

    def menu_save_presets(self):
        for i in range(len(self.dcs)):
            self.update_ui_dc(i)
            self.menu_save_preset()

    def menu_save_azavg_only(self):
        if self.dcs[self.current_page].azavg is not None:
            file.save_azavg_only(self.dcs[self.current_page].azavg)

    def btn_page_right_clicked(self):
        if not self.current_page == len(self.dcs) - 1:
            self.update_ui_dc(self.current_page + 1)

    def btn_page_left_clicked(self):
        if not self.current_page == 0:
            self.update_ui_dc(self.current_page - 1)

    def update_ui_dc(self,i):
        if not len(self.dcs) == 1 :
            self.dcs[self.current_page].release()
        self.current_page = i
        self.dcs[self.current_page].image_ready()

        # update number
        self.imgPanel.lbl_current_num.setText(str(self.current_page + 1) + "/" + str(len(self.dcs)))

        # update img
        self.update_img()

        # update graph
        self.update_azavg_graph()

        # update spinbox and settings
        self.update_center_spinBox()

        # windows title : file name
        if self.dcs[self.current_page].load_file_path is not None:
            self.setWindowTitle(self.dcs[self.current_page].load_file_path)
            fn = os.path.split(self.dcs[self.current_page].load_file_path)[1]
            max_size = 25
            if len(fn) > max_size:
                self.controlPanel.openFilePanel.lbl_file_name.setText("..."+fn[-max_size:])
            else:
                self.controlPanel.openFilePanel.lbl_file_name.setText(fn)

    def update_center_spinBox(self):
        if not self.dcs[self.current_page].img is None:
            self.controlPanel.settingPanel.spinBox_center_x.setMaximum(self.dcs[self.current_page].img.shape[0])  # todo : confusing x,y
            self.controlPanel.settingPanel.spinBox_center_y.setMaximum(self.dcs[self.current_page].img.shape[1])

        if not self.dcs[self.current_page].center[0] is None:
            ui_util.update_value(self.controlPanel.settingPanel.spinBox_center_x, self.dcs[self.current_page].center[0])
            ui_util.update_value(self.controlPanel.settingPanel.spinBox_center_y, self.dcs[self.current_page].center[1])

    def update_img(self):
        if self.dcs[self.current_page].img is None:
            self.imgPanel.update_img(np.zeros([1,1]))
            return
        img = self.dcs[self.current_page].img.copy()
        if self.controlPanel.settingPanel.chkBox_show_beam_stopper_mask.isChecked():
            img = cv2.bitwise_and(img, img, mask=np.bitwise_not(image_process.mask))
        if self.dcs[self.current_page].center[0] is not None and self.controlPanel.settingPanel.chkBox_show_centerLine.isChecked():
            img = image_process.draw_center_line(img, self.dcs[self.current_page].center)
        self.imgPanel.update_img(img)

    def dialog_to_range(self):
        left = int(self.graphPanel.spinBox_pixel_range_left.value())
        right = int(self.graphPanel.spinBox_pixel_range_right.value())
        ui_util.update_value(self.graphPanel.region,[left,right])
        self.dcs[self.current_page].pixel_start_n = left
        self.dcs[self.current_page].pixel_end_n = right
        if self.dcs[self.current_page].analyser is not None:
            # self.dcs[self.current_page].pixel_start_n = int(left)
            # self.dcs[self.current_page].pixel_end_n = int(right)
            self.dcs[self.current_page].analyser.instantfit()

    def range_to_dialog(self):
        left, right = self.graphPanel.region.getRegion()
        left = int(np.round(left))
        right = int(np.round(right))
        if right > len(self.dcs[self.current_page].azavg)-1:
            right = len(self.dcs[self.current_page].azavg)-1
        if left < 0:
            left = 0
        ui_util.update_value(self.graphPanel.region,[left, right])
        ui_util.update_value(self.graphPanel.spinBox_pixel_range_left,left)
        ui_util.update_value(self.graphPanel.spinBox_pixel_range_right,right)
        self.dcs[self.current_page].pixel_start_n = left
        self.dcs[self.current_page].pixel_end_n = right
        if self.dcs[self.current_page].analyser is not None:
            self.dcs[self.current_page].analyser.instantfit()


class ControlPanel(QtWidgets.QWidget):
    # text_fixed_height = 25
    # text_fixed_width = 70

    def __init__(self, mainWindow):
        QtWidgets.QWidget.__init__(self)
        self.openFilePanel = self.OpenFilePanel("OpenFile", mainWindow)
        self.settingPanel = self.SettingPanel("Settings")
        self.operationPanel = self.OperationPanel("Operation")

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.openFilePanel)
        layout.addWidget(self.settingPanel)
        layout.addWidget(self.operationPanel)
        layout.addStretch(1)
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
            layout = QtWidgets.QHBoxLayout()
            self.menu_file = self.create_menu(mainWindow)
            self.lbl_file_name = QtWidgets.QLabel("...")
            layout.addWidget(self.menu_file)
            layout.addWidget(self.lbl_file_name)
            self.setLayout(layout)

        def create_menu(self, mainWindow: QtWidgets.QMainWindow):
            menubar = mainWindow.menuBar()
            menubar.setNativeMenuBar(False)

            self.open_img_file = QtWidgets.QAction("Open &image file", self)
            self.open_img_folder = QtWidgets.QAction("Open &image folder", self)
            self.open_preset = QtWidgets.QAction("Open &preset", self)
            self.save_preset = QtWidgets.QAction("Save &preset", self)
            self.open_presets = QtWidgets.QAction("Open p&resets", self)
            self.save_presets = QtWidgets.QAction("Save p&resets", self)
            self.open_azavg_only = QtWidgets.QAction("Open &azavg only", self)
            self.save_azavg_only = QtWidgets.QAction("Save &azavg only", self)

            filemenu = menubar.addMenu("     File     ")
            filemenu.addAction(self.open_img_file)
            filemenu.addAction(self.open_img_folder)
            filemenu.addSeparator()
            filemenu.addAction(self.open_preset)
            filemenu.addAction(self.save_preset)
            filemenu.addSeparator()
            filemenu.addAction(self.open_presets)
            filemenu.addAction(self.save_presets)
            filemenu.addSeparator()
            filemenu.addAction(self.open_azavg_only)
            filemenu.addAction(self.save_azavg_only)

            menubar.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            return menubar

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


            # lbl_pixel_range = QtWidgets.QLabel("pixel Range")
            # self.spinBox_pixel_range_left = QtWidgets.QSpinBox()
            # self.spinBox_pixel_range_right = QtWidgets.QSpinBox()

            # layout.addWidget(lbl_pixel_range, 6, 0, 1, 2)
            # layout.addWidget(self.spinBox_pixel_range_left, 6, 2)
            # layout.addWidget(self.spinBox_pixel_range_right, 6, 3)

            self.setLayout(layout)

    class OperationPanel(QtWidgets.QGroupBox):
        def __init__(self,arg):
            QtWidgets.QGroupBox.__init__(self,arg)
            layout = QtWidgets.QGridLayout()
            self.btn_find_center = QtWidgets.QPushButton("Find center")
            self.btn_get_azimuthal_avg = QtWidgets.QPushButton("Get azimuthal data")
            self.btn_calculate_all_azimuthal = QtWidgets.QPushButton("Calculate all data")
            self.progress_bar = QtWidgets.QProgressBar()
            self.progress_bar.setValue(0)
            self.btn_open_epdf_analyser = QtWidgets.QPushButton("Open pdf analyser")

            layout.addWidget(self.btn_find_center, 0, 0)
            layout.addWidget(self.btn_get_azimuthal_avg, 0, 1)
            # layout.addWidget(self.btn_save_current_azimuthal, 1, 0,1,2)
            layout.addWidget(self.btn_calculate_all_azimuthal, 2, 0, 1, 2)
            layout.addWidget(self.progress_bar,3,0,1,2)
            layout.addWidget(self.btn_open_epdf_analyser,4,0,1,2)

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
        self.setMinimumWidth(300)
        self.setMinimumHeight(300)
        self.setLayout(layout)
        self._current_data = None
        self.cmap = pg.ColorMap(np.linspace(0, 1, len(image_process.colorcube)), color=image_process.colorcube)
    def update_img(self,img):
        self.imageView.setColorMap(self.cmap)
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
        # self.plot_azav = pg.PlotWidget(title='azimuthal average')
        self.plot_azav = ui_util.IntensityPlotWidget(title='azimuthal average')
        self.plot_azav.setYScaling(True)
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.plot_azav)

        self.setLayout(self.layout)
        self.setMinimumHeight(200)
        self.region = pg.LinearRegionItem([0, 100])
        self.plot_azav.region = self.region
        self.plot_azav.addItem(self.region)

        self.legend = self.plot_azav.addLegend(offset=(-30,30))

        self.button_grp_widget = QtWidgets.QWidget()
        self.button_grp_widget.layout = QtWidgets.QVBoxLayout()
        self.button_grp_widget.setLayout(self.button_grp_widget.layout)

        self.button_grp_widget.layout.addStretch(1)
        lbl_pixel_range = QtWidgets.QLabel("pixel Range")
        self.spinBox_pixel_range_left = QtWidgets.QSpinBox()
        self.spinBox_pixel_range_right = QtWidgets.QSpinBox()
        self.button_grp_widget.layout.addWidget(lbl_pixel_range)
        self.button_grp_widget.layout.addWidget(self.spinBox_pixel_range_left)
        self.button_grp_widget.layout.addWidget(self.spinBox_pixel_range_right)

        
        self.button_start = QtWidgets.QPushButton("Start")
        self.button_all = QtWidgets.QPushButton("All")
        self.button_end = QtWidgets.QPushButton("End")
        self.button_select = QtWidgets.QPushButton("Select")
        self.button_grp_widget.layout.addWidget(self.button_start)
        self.button_grp_widget.layout.addWidget(self.button_all)
        self.button_grp_widget.layout.addWidget(self.button_end)
        self.button_grp_widget.layout.addWidget(self.button_select)
        self.button_grp_widget.layout.addStretch(1)

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
        else:
            self.azav_prev_dat = self.azav_curr_dat
            self.azav_curr_dat = dat
            self.plot_azav_curr.setData(self.azav_curr_dat)
            self.plot_azav_prev.setData(self.azav_prev_dat)

        # self.plotWindow.resize(1000,350)




if __name__ == '__main__':
    qtapp = QtWidgets.QApplication.instance()
    if not qtapp:
        qtapp = QtWidgets.QApplication(sys.argv)
    app = DataViewer()
    app.show()
    sys.exit(qtapp.exec_())