from PyQt6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np
import util
import time
from calculate import image_process, polar_transform, elliptical_correction
from ui import ui_util
pg.setConfigOptions(antialias=True)
import definitions
import cv2
from datacube.cube import PDFCube
from ui.roi_selector import mask_module
from file import load


class ProfileExtraction(QtWidgets.QWidget):
    def __init__(self, Dataviewer):
        QtWidgets.QWidget.__init__(self)
        self.Dataviewer = Dataviewer
        self.mask_module = mask_module.MaskModule(fp=definitions.MASK_PATH)
        self.init_ui()
        self.isShowCenter = True
        self.flag_range_update = False
        self.sig_binding()
        self.dc:PDFCube = None



        # for text
        # self.menu_open_azavg_only(np.loadtxt("/mnt/experiment/TEM diffraction/201126 (test)/sample38_TiTa_annealed/Analysis ePDFpy/Camera 230 mm Ceta 20201126 1649_40s_20f_area01.azavg.txt"))
        # self.menu_open_azavg_only(np.loadtxt(r"V:\experiment\TEM diffraction\201126 (test)\sample38_TiTa_annealed\Analysis ePDFpy\Camera 230 mm Ceta 20201126 1649_40s_20f_area01.azavg.txt"))
        ##

    def init_ui(self):
        self.control_panel = ControlPanel(self.Dataviewer, self)
        self.img_panel = ImgPanel()
        self.profile_graph_panel = IntensityProfilePanel()
        self.polar_image_panel = PolarImagePanel()

        self.upper_left = self.control_panel
        self.bottom_left = self.img_panel
        self.upper_right = self.profile_graph_panel
        self.bottom_right = self.polar_image_panel

        self.splitter_left_vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.splitter_left_vertical.addWidget(self.upper_left)
        self.splitter_left_vertical.addWidget(self.bottom_left)
        self.splitter_left_vertical.setStretchFactor(1, 1)

        self.splitter_right_vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.splitter_right_vertical.addWidget(self.upper_right)
        self.splitter_right_vertical.addWidget(self.bottom_right)
        # self.splitter_right_vertical.setStretchFactor(0, 1)
        self.splitter_right_vertical.setStretchFactor(1, 1)

        self.left = self.splitter_left_vertical
        self.right = self.splitter_right_vertical

        self.splitter_horizontal = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.splitter_horizontal.addWidget(self.left)
        self.splitter_horizontal.addWidget(self.right)
        self.splitter_horizontal.setStretchFactor(0, 10)
        self.splitter_horizontal.setStretchFactor(1, 7)

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(3, 3, 3, 3)
        self.layout.addWidget(self.splitter_horizontal)

        self.setLayout(self.layout)


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
        # self.control_panel.ellipticalCorrectionPanel.btn_fit.clicked.connect(self.elliptical_correction)
        self.control_panel.settingPanel.spinBox_center_x.valueChanged.connect(self.spinbox_changed_event)
        self.control_panel.settingPanel.spinBox_center_y.valueChanged.connect(self.spinbox_changed_event)
        # self.control_panel.ellipticalCorrectionPanel.spinBox_a.valueChanged.connect(self.spinbox_changed_event)
        # self.control_panel.ellipticalCorrectionPanel.spinBox_b.valueChanged.connect(self.spinbox_changed_event)
        # self.control_panel.ellipticalCorrectionPanel.spinBox_theta.valueChanged.connect(self.spinbox_changed_event)
        self.control_panel.operationPanel.btn_calculate_all_azimuthal.clicked.connect(self.calculate_all_azimuthal)
        self.control_panel.settingPanel.chkBox_show_centerLine.stateChanged.connect(self.update_img)
        self.control_panel.settingPanel.chkBox_show_beam_stopper_mask.stateChanged.connect(self.update_img)

        self.control_panel.saveLoadPanel.open_img_file.triggered.connect(self.Dataviewer.menu_open_image_file)
        self.control_panel.saveLoadPanel.open_img_stack_mrc.triggered.connect(lambda: self.Dataviewer.menu_open_image_stack('.mrc'))
        self.control_panel.saveLoadPanel.open_img_stack_csv.triggered.connect(lambda: self.Dataviewer.menu_open_image_stack('.csv'))
        self.control_panel.saveLoadPanel.open_img_stack_tiff.triggered.connect(lambda: self.Dataviewer.menu_open_image_stack('.tiff'))
        self.control_panel.saveLoadPanel.open_img_stack_png.triggered.connect(lambda: self.Dataviewer.menu_open_image_stack('.png'))
        self.control_panel.saveLoadPanel.open_img_stack_txt.triggered.connect(lambda: self.Dataviewer.menu_open_image_stack('.txt'))
        self.control_panel.saveLoadPanel.open_img_stack_custom.triggered.connect(lambda: self.Dataviewer.menu_open_image_stack('.custom'))
        self.control_panel.saveLoadPanel.open_img_stack_jpg.triggered.connect(lambda: self.Dataviewer.menu_open_image_stack('.jpg'))
        self.control_panel.saveLoadPanel.open_img_stack_jpeg.triggered.connect(lambda: self.Dataviewer.menu_open_image_stack('.jpeg'))

        self.control_panel.saveLoadPanel.save_current_azavg.triggered.connect(self.Dataviewer.menu_save_azavg_only)
        self.control_panel.saveLoadPanel.save_azavg_stack.triggered.connect(self.Dataviewer.menu_save_azavg_stack)

        self.mask_module.mask_changed.connect(self.update_img)
        self.control_panel.blankimagePanel.imagebutton.clicked.connect(self.imagebuttonClicked)   # MH edit 2023/07/05
        self.control_panel.blankimagePanel.removebutton.clicked.connect(self.btn_blank_remove)    # MH edit 2023/07/05
        self.control_panel.blankimagePanel.revertbutton.clicked.connect(self.btn_blank_revert)    # MH edit 2023/07/05
        # self.control_panel.blankimagePanel.noiseremove.clicked.connect(self.blank_remove)
        # self.control_panel.ellipticalCorrectionPanel.chkbox_use_elliptical_correction.stateChanged.connect(self.use_elliptical_correction)

    # def use_elliptical_correction(self, state):
    #     self.control_panel.ellipticalCorrectionPanel.spinBox_a.setEnabled(state)
    #     self.control_panel.ellipticalCorrectionPanel.spinBox_b.setEnabled(state)
    #     self.control_panel.ellipticalCorrectionPanel.spinBox_theta.setEnabled(state)
    #     self.control_panel.ellipticalCorrectionPanel.btn_fit.setEnabled(state)
    #
    # def elliptical_correction(self):
    #     self.dc.elliptical_fitting()
    #     a,b,theta = self.dc.p_ellipse
    #     ui_util.update_value(self.control_panel.ellipticalCorrectionPanel.spinBox_a, a)
    #     ui_util.update_value(self.control_panel.ellipticalCorrectionPanel.spinBox_b, b)
    #     ui_util.update_value(self.control_panel.ellipticalCorrectionPanel.spinBox_theta, theta)


    def spinbox_changed_event(self):
        x = self.control_panel.settingPanel.spinBox_center_x.value()
        y = self.control_panel.settingPanel.spinBox_center_y.value()
        # a = self.control_panel.ellipticalCorrectionPanel.spinBox_a.value()           # MH edit 2023/07/06
        # b = self.control_panel.ellipticalCorrectionPanel.spinBox_b.value()           # MH edit 2023/07/06
        # theta = self.control_panel.ellipticalCorrectionPanel.spinBox_theta.value()   # MH edit 2023/07/06
        a = 1
        b = 1
        theta = 0
        # print("spinbox changed, ", x, y, a, b, theta)                                # MH edit 2023/07/06
        print("spinbox changed, ", x, y)
        self.dc.center = (x, y)
        self.dc.p_ellipse = (a,b,theta)
        self.update_img()

    def calculate_all_azimuthal(self):
        tic = time.time()
        print(self.Dataviewer.dcs)
        for i in range(len(self.Dataviewer.dcs)):
            # print("processing azimuthal values", self.dc.img_file_path)
            self.Dataviewer.load_dc(i)
            # if self.Dataviewer.dcs[i].img is not None and self.Dataviewer.dcs[i].center[0] is None:
            if self.Dataviewer.dcs[i].data is not None and self.Dataviewer.dcs[i].center[0] is None:  # edited 01/17
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
        self.dc.mask = self.mask_module.get_current_mask()
        self.dc.calculate_azimuthal_average()

        # update ui
        self.update_azavg_graph()
        self.Dataviewer.PDF_analyser.update_initial_iq()
        self.Dataviewer.PDF_analyser.update_initial_iq_graph()
        self.update_center_spinBox()
        self.update_img()
        # self.update_polar_img()   # MH edit 2023/07/06

    def update_azavg_graph(self):
        if self.dc.azavg is None:
            return
        self.profile_graph_panel.update_graph(self.dc.azavg)

    def update_polar_img(self):
        if self.dc.center[0] is None or self.dc.data is None:
            return
        # MH edit 2023/07/04 -> Remove elliptical fit
        # if self.control_panel.ellipticalCorrectionPanel.chkbox_use_elliptical_correction.isChecked() and self.dc.p_ellipse is not None:
        #     p_ellipse = self.dc.p_ellipse
        else:
            p_ellipse = [1,1,0]
        self.dc.p_ellipse = p_ellipse
        # polar_img = self.dc.elliptical_transformation(dphi=np.radians(0.5))
        polar_img, _, _ = polar_transform.cartesian_to_polarelliptical_transform(self.dc.data,[self.dc.center[1],self.dc.center[0],p_ellipse[0],p_ellipse[1],p_ellipse[2]], mask = ~self.mask_module.mask, dphi=np.radians(0.5))
        self.polar_image_panel.update_img(polar_img)

    def find_center(self):
        i1 = self.control_panel.settingPanel.spinBox_irange1.value()
        i2 = self.control_panel.settingPanel.spinBox_irange2.value()
        intensity_range = (i1, i2)
        slice_count = int(self.control_panel.settingPanel.spinBox_slice_count.value())
        mask = self.mask_module.get_current_mask()
        if mask is not None:
            self.dc.mask = ~mask # todo: mask 방향정리
        self.dc.find_center()
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
        self.update_polar_img()

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
        self.control_panel.blankimagePanel.label.clear()
        # if self.currentblankpath is None:
        # if self.dc.origindata is None:
        if self.dc.origindata is None:
            return
        if self.dc.blankpath is not None and (self.dc.data == self.dc.origindata).all() == False:
            self.control_panel.blankimagePanel.label.setText(self.dc.blankpath[self.dc.blankpath.rfind('/') + 1:] + ' Removed')
        if self.dc.blankpath is not None and (self.dc.data == self.dc.origindata).all() == True:
            self.control_panel.blankimagePanel.label.setText(self.dc.blankpath[self.dc.blankpath.rfind('/') + 1:] + ' Opened')

    def update_center_spinBox(self):
        if not self.dc.data is None:
            self.control_panel.settingPanel.spinBox_center_x.setMaximum(self.dc.data.shape[0])  # todo : confusing x,y
            self.control_panel.settingPanel.spinBox_center_y.setMaximum(self.dc.data.shape[1])

        if not self.dc.center[0] is None:
            ui_util.update_value(self.control_panel.settingPanel.spinBox_center_x, self.dc.center[0])
            ui_util.update_value(self.control_panel.settingPanel.spinBox_center_y, self.dc.center[1])

    def update_img(self):
        if not hasattr(self,'dc') or self.dc is None:
            return
        if self.dc.data is None:
            self.img_panel.clear_img()
            self.polar_image_panel.imageView.clear()
            return
        img = self.dc.img_display.copy()
        if self.control_panel.settingPanel.chkBox_show_beam_stopper_mask.isChecked():
            if self.mask_module.mask is not None:
                img = cv2.bitwise_and(img, img, mask=~self.mask_module.mask)
            # img = cv2.bitwise_and(img, img, mask=np.bitwise_not(self.mask_module.mask))
        if self.dc.center[0] is not None and self.control_panel.settingPanel.chkBox_show_centerLine.isChecked():
            img = image_process.draw_center_line(img, self.dc.center)
        self.img_panel.update_img(img)
        self.update_polar_img()

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

    def imagebuttonClicked(self): # MH edit 2023/07/05
        blankfp,_ = QtWidgets.QFileDialog.getOpenFileNames(None)
        blankimg = load.load_diffraction_image(blankfp[0])
        self.currentblankpath = blankfp[0]
        self.control_panel.blankimagePanel.label.setText(self.currentblankpath[self.currentblankpath.rfind('/')+1:] + ' Opened')
        assert self.dc.img_display.shape == blankimg.shape, "Check Image Shape!"
        self.dc.blank, self.dc.blankpath, self.currentnoise = blankimg, self.currentblankpath, blankimg.copy()

    def btn_blank_remove(self):
        reply = QtWidgets.QMessageBox.question(self,'Remove noise',
                                               'Remove noise from the data?\n\nYes: Current data\n Yes to all: All data',
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.YesToAll | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Yes)
        if reply == QtWidgets.QMessageBox.Yes:
            # self.dc.blankpath = self.currentblankpath
            self.dc.apply_blank()
            self.update_dc(self.dc)

        if reply == QtWidgets.QMessageBox.YesToAll:
            self.currentnoise, self.currentblankpath = self.dc.blank, self.dc.blankpath
            for i in range(len(self.Dataviewer.dcs)):
                # print("processing azimuthal values", self.dc.img_file_path)
                self.Dataviewer.load_dc(i)
                # if self.Dataviewer.dcs[i].img is not None and self.Dataviewer.dcs[i].center[0] is None:
                if self.Dataviewer.dcs[i].data is not None:  # edited 2023/07/05
                    self.Dataviewer.dcs[i].blank, self.Dataviewer.dcs[i].blankpath = self.currentnoise, self.currentblankpath
                    self.Dataviewer.dcs[i].apply_blank()
                    self.update_dc(self.Dataviewer.dcs[i])
                print("({}/{}) Noise subtraction applied".format(i + 1, len(self.Dataviewer.dcs)))

    def btn_blank_revert(self):
        reply = QtWidgets.QMessageBox.question(self,'Revert noise',
                                               'Revert noise from the data?\n\nYes: Current data\n Yes to all: All data',
                                                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.YesToAll | QtWidgets.QMessageBox.Cancel, QtWidgets.QMessageBox.Yes)
        if reply == QtWidgets.QMessageBox.Yes:
            self.dc.remove_blank()
            self.update_dc(self.dc)

        if reply == QtWidgets.QMessageBox.YesToAll:
            for i in range(len(self.Dataviewer.dcs)):
                # print("processing azimuthal values", self.dc.img_file_path)
                self.Dataviewer.load_dc(i)
                # if self.Dataviewer.dcs[i].img is not None and self.Dataviewer.dcs[i].center[0] is None:
                if self.Dataviewer.dcs[i].data is not None:  # edited 2023/07/05
                    self.Dataviewer.dcs[i].remove_blank()
                    self.update_dc(self.Dataviewer.dcs[i])
                print("({}/{}) Noise subtraction reverted".format(i + 1, len(self.Dataviewer.dcs)))


    def blank_apply(self):
        self.dc.blankpath = self.currentblankpath
        self.dc.apply_blank()
        self.update_dc(self.dc)

        # self.control_panel.blankimagePanel.label.setText(self.dc.blankpath[self.dc.blankpath.rfind('/') + 1:])
    def blank_apply_all(self):
        for i in range(len(self.Dataviewer.dcs)):
            # print("processing azimuthal values", self.dc.img_file_path)
            self.Dataviewer.load_dc(i)
            # if self.Dataviewer.dcs[i].img is not None and self.Dataviewer.dcs[i].center[0] is None:
            if self.Dataviewer.dcs[i].data is not None:  # edited 2023/07/05
                self.Dataviewer.dcs[i].blank, self.Dataviewer.dcs[i].blankpath = self.currentnoise, self.currentblankpath
                self.Dataviewer.dcs[i].apply_blank()
                self.update_dc(self.Dataviewer.dcs[i])
            print("({}/{}) Noise subtraction applied".format(i+1,len(self.Dataviewer.dcs)))

    def blank_remove(self):
        self.dc.remove_blank()
        self.update_dc(self.dc)
class ControlPanel(QtWidgets.QWidget):
    # text_fixed_height = 25
    # text_fixed_width = 70

    def __init__(self, mainWindow, profile_extraction):
        QtWidgets.QWidget.__init__(self)
        # self.openFilePanel = self.OpenFilePanel("OpenFile", mainWindow)
        self.temp_layout = QtWidgets.QVBoxLayout()
        self.temp_layout2 = QtWidgets.QHBoxLayout()
        self.settingPanel = self.SettingPanel("Center finding setting")
        self.operationPanel = self.OperationPanel("Operation")
        # self.ellipticalCorrectionPanel = self.EllipticalCorrectionPanel("Elliptical Correction")
        self.saveLoadPanel = self.SaveLoadPanel("Save and Load",mainWindow)
        self.maskPanel = self.MaskModule("Mask", profile_extraction)
        self.blankimagePanel = self.BlankImagePanel("Background noise")  # MH edit 2023/07/05
        self.temp_layout2.addWidget(self.saveLoadPanel)
        self.temp_layout2.addWidget(self.maskPanel)
        self.temp_layout.addLayout(self.temp_layout2)
        self.temp_layout.addWidget(self.settingPanel)


        layout = QtWidgets.QHBoxLayout()
        # layout.addWidget(self.openFilePanel)
        layout.addLayout(self.temp_layout)
        layout.addWidget(self.blankimagePanel)
        layout.addWidget(self.operationPanel)
        self.operationPanel.setFixedWidth(200)
        # layout.addWidget(self.ellipticalCorrectionPanel)
        self.setLayout(layout)


    class BlankImagePanel(QtWidgets.QGroupBox):   # MH edit 2023/07/05
        def __init__(self, arg):
            QtWidgets.QGroupBox.__init__(self,arg)
            self.layout = QtWidgets.QVBoxLayout()
            self.imagebutton = QtWidgets.QPushButton("Open Blank Image")
            self.removebutton = QtWidgets.QPushButton("Remove noise")
            self.revertbutton = QtWidgets.QPushButton("Revert noise")
            # self.noiseremove = QtWidgets.QPushButton("Remove noise")
            self.label = QtWidgets.QLabel()

            self.layout.addWidget(self.imagebutton)
            self.layout.addWidget(self.removebutton)
            self.layout.addWidget(self.revertbutton)
            # self.layout.addWidget(self.noiseremove)
            self.layout.addWidget(self.label)
            self.setLayout(self.layout)


    class MaskModule(QtWidgets.QGroupBox):
        def __init__(self, arg, profile_extraction):
            QtWidgets.QGroupBox.__init__(self, arg)
            self.layout = QtWidgets.QHBoxLayout()
            self.setLayout(self.layout)
            self.mask_dropdown = profile_extraction.mask_module.dropdown
            # self.mask_dropdown.setMinimumWidth(100)
            self.layout.addWidget(self.mask_dropdown)

    class SaveLoadPanel(QtWidgets.QGroupBox):
        def __init__(self, arg, mainWindow:QtWidgets.QMainWindow):
            QtWidgets.QGroupBox.__init__(self, arg)
            menubar = mainWindow.menuBar()
            menubar.setNativeMenuBar(False)
            open_menu = menubar.addMenu("&Open")
            save_menu = menubar.addMenu("&Save")

            self.open_img_file = QtGui.QAction("Open &image file", self)
            open_menu.addAction(self.open_img_file)
            self.open_img_stack = open_menu.addMenu("Open image stack")
            self.open_img_stack_mrc = QtGui.QAction("mrc file stack", self)
            self.open_img_stack_txt = QtGui.QAction("txt file stack", self)
            self.open_img_stack_csv = QtGui.QAction("csv file stack", self)
            self.open_img_stack_tiff = QtGui.QAction("tiff file stack", self)
            self.open_img_stack_jpg = QtGui.QAction("jpg file stack", self)
            self.open_img_stack_jpeg = QtGui.QAction("jpeg file stack", self)
            self.open_img_stack_png = QtGui.QAction("png file stack", self)
            self.open_img_stack_custom = QtGui.QAction("Custom ...", self)
            self.open_img_stack.addAction(self.open_img_stack_mrc)
            self.open_img_stack.addAction(self.open_img_stack_txt)
            self.open_img_stack.addAction(self.open_img_stack_csv)
            self.open_img_stack.addAction(self.open_img_stack_tiff)
            self.open_img_stack.addAction(self.open_img_stack_jpg)
            self.open_img_stack.addAction(self.open_img_stack_jpeg)
            self.open_img_stack.addAction(self.open_img_stack_png)
            self.open_img_stack.addAction(self.open_img_stack_custom)

            self.save_current_azavg = QtGui.QAction("Save current azavg file", self)
            self.save_azavg_stack = QtGui.QAction("Save azavg stack", self)
            save_menu.addAction(self.save_current_azavg)
            save_menu.addAction(self.save_azavg_stack)

            self.layout = QtWidgets.QHBoxLayout()
            self.setLayout(self.layout)
            self.layout.addWidget(menubar)

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
            # layout.addWidget(lbl_intensity_range, 1, 0, 1, 2)
            # layout.addWidget(lbl_slice_count, 2, 0, 1, 2)
            # layout.addWidget(self.spinBox_irange1, 1, 2)
            # layout.addWidget(self.spinBox_irange2, 1, 3)
            # layout.addWidget(self.spinBox_slice_count, 2, 2)
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

            self.btn_find_center = QtWidgets.QPushButton("Find center")
            self.btn_get_azimuthal_avg = QtWidgets.QPushButton("Get azimuthal data")
            self.btn_calculate_all_azimuthal = QtWidgets.QPushButton("Calculate all data")
            self.progress_bar = QtWidgets.QProgressBar()
            self.progress_bar.setValue(0)

            # layout = QtWidgets.QGridLayout()
            # layout.addWidget(self.btn_find_center, 0, 0)
            # layout.addWidget(self.btn_get_azimuthal_avg, 0, 1)
            # # layout.addWidget(self.btn_save_current_azimuthal, 1, 0,1,2)
            # layout.addWidget(self.btn_calculate_all_azimuthal, 2, 0, 1, 2)
            # layout.addWidget(self.progress_bar, 3, 0, 1, 2)
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(self.btn_find_center)
            layout.addWidget(self.btn_get_azimuthal_avg)
            layout.addWidget(self.btn_calculate_all_azimuthal)
            layout.addWidget(self.progress_bar)

            self.setLayout(layout)

    class EllipticalCorrectionPanel(QtWidgets.QGroupBox):
        def __init__(self, arg):
            QtWidgets.QGroupBox.__init__(self, arg)
            layout = QtWidgets.QVBoxLayout()
            self.chkbox_use_elliptical_correction = QtWidgets.QCheckBox("Use elliptical correction")
            self.chkbox_use_elliptical_correction.setChecked(True)

            self.lbl_a = QtWidgets.QLabel("a")
            self.lbl_b = QtWidgets.QLabel("b")
            self.lbl_theta = QtWidgets.QLabel("θ")

            self.spinBox_a = QtWidgets.QDoubleSpinBox()
            self.spinBox_a.setValue(1)
            self.spinBox_a.setMaximum(10e6)
            self.spinBox_a.setSingleStep(0.01)
            layout_a = QtWidgets.QHBoxLayout()
            layout_a.addWidget(self.lbl_a)
            layout_a.addWidget(self.spinBox_a)

            self.spinBox_b = QtWidgets.QDoubleSpinBox()
            self.spinBox_b.setValue(1)
            self.spinBox_b.setMaximum(10e6)
            self.spinBox_b.setSingleStep(0.01)
            layout_b = QtWidgets.QHBoxLayout()
            layout_b.addWidget(self.lbl_b)
            layout_b.addWidget(self.spinBox_b)

            self.spinBox_theta = QtWidgets.QDoubleSpinBox()
            self.spinBox_theta.setValue(0)
            self.spinBox_theta.setMaximum(10e6)
            self.spinBox_theta.setSingleStep(0.1)
            layout_theta = QtWidgets.QHBoxLayout()
            layout_theta.addWidget(self.lbl_theta)
            layout_theta.addWidget(self.spinBox_theta)

            self.btn_fit = QtWidgets.QPushButton("Fit")

            layout.addWidget(self.chkbox_use_elliptical_correction)
            layout.addWidget(self.btn_fit)
            layout.addLayout(layout_a)
            layout.addLayout(layout_b)
            layout.addLayout(layout_theta)
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
        self.imageView.setColorMap(self.cmap)

    def update_img(self, img):
        self._current_data = img
        if len(img.shape) == 2:
            self.imageView.setImage(self._current_data.transpose(1, 0), autoRange=False)
        if len(img.shape) == 3:
            self.imageView.setImage(self._current_data.transpose(1, 0, 2), autoRange=False)

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

class PolarImagePanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        plot = pg.PlotItem()
        plot.setLabel(axis='left')
        plot.setLabel(axis='bottom')

        self.imageView = pg.ImageView(view=plot)
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.imageView)
        self.setLayout(self.layout)
        self.cmap = pg.ColorMap(np.linspace(0, 1, len(image_process.colorcube)), color=image_process.colorcube)
        self.imageView.setColorMap(self.cmap)

        show_histogram = False
        if show_histogram:
            self.imageView.ui.histogram.show()
            self.imageView.ui.roiBtn.show()
            self.imageView.ui.menuBtn.show()
        else:
            self.imageView.ui.histogram.hide()
            self.imageView.ui.roiBtn.hide()
            self.imageView.ui.menuBtn.hide()

    def update_img(self, img):
        self._current_data = img
        if len(img.shape) == 2:
            self.imageView.setImage(self._current_data.transpose(1, 0), autoRange=False)
        if len(img.shape) == 3:
            self.imageView.setImage(self._current_data.transpose(1, 0, 2), autoRange=False)

    def clear_img(self):
        self.imageView.clear()

    def get_img(self):
        return self._current_data