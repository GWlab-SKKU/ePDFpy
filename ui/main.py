from PyQt5 import QtCore, QtWidgets, QtGui
import sys
import pyqtgraph as pg
import file
import util
from datacube import DataCube
from typing import List
from ui.pdfanalysis import PdfAnalysis
from PyQt5.QtWidgets import QMessageBox
import ui.selection_analysis.averaging_multiple_gr as averaging_multiple_gr
from ui import ui_util

pg.setConfigOptions(antialias=True)
import definitions
from ui.profile_extraction import ProfileExtraction

class DataViewer(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.top_menu = self.TopMenu(self)
        self.bottom = QtWidgets.QTabWidget()

        self.profile_extraction = ProfileExtraction(self)
        self.PDF_analyser = PdfAnalysis(self)

        self.bottom.addTab(self.profile_extraction,"Profile extraction")
        self.bottom.addTab(self.PDF_analyser, "PDF analysis")

        self.dcs: List[DataCube] = []

        self.setStyleSheet(ui_util.get_style_sheet())

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.top_menu)
        layout.addWidget(self.bottom)
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)

        centralWidget = QtWidgets.QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)
        self.setWindowTitle(definitions.PROGRAM_NAME)
        self.sig_binding()
        self.resize(1300,800)

    class TopMenu(QtWidgets.QWidget):
        def __init__(self, mainWindow):
            self.mainWindow = mainWindow
            QtWidgets.QWidget.__init__(self)
            layout = QtWidgets.QGridLayout()
            left_section = self.create_menu(mainWindow)
            # left_section = QtWidgets.QWidget()
            center_section = self.create_navigator()
            right_section = self.create_data_quality()
            layout.addWidget(left_section)
            # layout.addWidget(center_section)
            # layout.addWidget(right_section)

            progress_bar = QtWidgets.QProgressBar()

            layout.addWidget(left_section,0,0,alignment=QtCore.Qt.AlignLeft)
            layout.addWidget(progress_bar,0,1,alignment=QtCore.Qt.AlignLeft)
            layout.addWidget(center_section,0,2,alignment=QtCore.Qt.AlignCenter)
            layout.addWidget(right_section,0,3,alignment=QtCore.Qt.AlignRight)

            # layout.addWidget(center_section)
            layout.setSpacing(0)
            layout.setContentsMargins(0,0,0,0)
            self.setLayout(layout)

        def create_menu(self, mainWindow: QtWidgets.QMainWindow):
            menubar = mainWindow.menuBar()
            menubar.setNativeMenuBar(False)
            # self.open_img_file = QtWidgets.QAction("Open &image file", self)
            # self.open_preset = QtWidgets.QAction("Open preset &file", self)
            # self.save_preset = QtWidgets.QAction("Save preset &file", self)
            # self.open_preset_stack = QtWidgets.QAction("Open preset &stack", self)
            # self.save_preset_stack = QtWidgets.QAction("Save preset &stack", self)
            # self.save_preset_option = QtWidgets.QAction("Save preset &option setting", self)
            # self.save_preset_option.setDisabled(True)
            # self.open_azavg_only = QtWidgets.QAction("Open &azavg only", self)
            # self.save_azavg_only = QtWidgets.QAction("Save &azavg only", self)
            self.averaging_gr = QtWidgets.QAction("Selection Analysis", self)
            #
            # open_menu = menubar.addMenu("     &Open     ")
            # open_menu.addAction(self.open_img_file)
            # self.open_img_stack = open_menu.addMenu("Open image stack")
            # # open_menu.addAction(self.open_img_stack)
            # open_menu.addSeparator()
            # open_menu.addAction(self.open_preset)
            # open_menu.addAction(self.open_preset_stack)
            # open_menu.addSeparator()
            # open_menu.addAction(self.open_azavg_only)
            # self.open_azavg_stack = open_menu.addMenu("Open azavg stack")
            #
            # self.open_img_stack_mrc = QtWidgets.QAction("mrc file stack", self)
            # self.open_img_stack_txt = QtWidgets.QAction("txt file stack", self)
            # self.open_img_stack_csv = QtWidgets.QAction("csv file stack", self)
            # self.open_img_stack_tiff = QtWidgets.QAction("tiff file stack", self)
            # self.open_img_stack_jpg = QtWidgets.QAction("jpg file stack", self)
            # self.open_img_stack_jpeg = QtWidgets.QAction("jpeg file stack", self)
            # self.open_img_stack_png = QtWidgets.QAction("png file stack", self)
            # self.open_img_stack_custom = QtWidgets.QAction("Custom ...", self)
            # self.open_img_stack.addAction(self.open_img_stack_mrc)
            # self.open_img_stack.addAction(self.open_img_stack_txt)
            # self.open_img_stack.addAction(self.open_img_stack_csv)
            # self.open_img_stack.addAction(self.open_img_stack_tiff)
            # self.open_img_stack.addAction(self.open_img_stack_jpg)
            # self.open_img_stack.addAction(self.open_img_stack_jpeg)
            # self.open_img_stack.addAction(self.open_img_stack_png)
            # self.open_img_stack.addAction(self.open_img_stack_custom)
            #
            # self.open_azavg_stack_csv = QtWidgets.QAction("csv", self)
            # self.open_azavg_stack_txt = QtWidgets.QAction("txt", self)
            # self.open_azavg_stack_azavg_txt = QtWidgets.QAction("azavg.txt", self)
            # self.open_azavg_stack_azavg_csv = QtWidgets.QAction("azavg.csv", self)
            # self.open_azavg_stack.addAction(self.open_azavg_stack_csv)
            # self.open_azavg_stack.addAction(self.open_azavg_stack_txt)
            # self.open_azavg_stack.addAction(self.open_azavg_stack_azavg_txt)
            # self.open_azavg_stack.addAction(self.open_azavg_stack_azavg_csv)
            #
            # save_menu = menubar.addMenu("     &Save     ")
            # save_menu.addAction(self.save_preset)
            # save_menu.addAction(self.save_preset_stack)
            # save_menu.addAction(self.save_preset_option)
            # save_menu.addSeparator()
            # save_menu.addAction(self.save_azavg_only)

            utility_menu = menubar.addMenu("     &Utility     ")
            utility_menu.addAction(self.averaging_gr)

            menubar.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            return menubar

        def create_tab(self):
            self.tab = self.PushTab()
            self.tab.add_button("Profile extraction")
            self.tab.add_button("PDF analysis")
            self.tab.btn_check(0)

            return self.tab

        class PushTab(QtWidgets.QWidget):
            def __init__(self):
                QtWidgets.QWidget.__init__(self)
                self.layout = QtWidgets.QHBoxLayout()
                self.setLayout(self.layout)
                self.button_list = []
                self.layout.setSpacing(0)
                self.layout.setContentsMargins(0,0,0,0)

            def add_button(self,title):
                button = QtWidgets.QPushButton(title)
                button.setCheckable(True)
                self.layout.addWidget(button)
                self.button_list.append(button)
                button.toggled.connect(lambda state, x=len(self.button_list)-1: self.btn_check(x))
                return button

            def btn_check(self, i):
                for j in range(len(self.button_list)):
                    if j == i:
                        # self.button_list[j].setChecked(True)
                        self.button_list[j].setEnabled(False)
                    else:
                        self.button_list[j].setChecked(False)
                        self.button_list[j].setEnabled(True)

        def create_navigator(self):
            self.navigator = QtWidgets.QWidget()
            self.navigator.layout = QtWidgets.QHBoxLayout()
            self.btn_left = QtWidgets.QPushButton("    <<    ")
            self.btn_right = QtWidgets.QPushButton("    >>    ")
            self.lbl_current_num = QtWidgets.QLabel("1")
            self.lbl_current_num.setAlignment(QtCore.Qt.AlignCenter)
            self.navigator.layout.addWidget(self.btn_left)
            self.navigator.layout.addWidget(self.lbl_current_num)
            self.navigator.layout.addWidget(self.btn_right)
            self.navigator.setLayout(self.navigator.layout)
            self.navigator.layout.setContentsMargins(0,0,0,0)
            self.navigator.layout.setSpacing(10)
            return self.navigator

        def create_data_quality(self):
            self.data_quality = QtWidgets.QWidget()
            self.data_quality.layout = QtWidgets.QHBoxLayout()

            self.lbl_data_quality = QtWidgets.QLabel("Data Quality")
            self.combo_dataQuality = QtWidgets.QComboBox()
            self.combo_dataQuality.setMinimumWidth(100)
            quality_list = ["None", "Auto"]
            quality_list.extend(util.df_data_quality['label'].to_list())  # [None,Auto,L1,L2,L3,L4]
            self.combo_dataQuality.addItems(quality_list)

            self.data_quality.layout.addWidget(self.lbl_data_quality)
            self.data_quality.layout.addWidget(self.combo_dataQuality)
            self.data_quality.setLayout(self.data_quality.layout)
            self.data_quality.layout.setContentsMargins(0,0,0,0)
            self.data_quality.layout.setSpacing(10)

            return self.data_quality

    def set_data_quality(self):
        # set auto(data quality) combolist and save to datacube
        txt_auto_quality = self.reload_auto_data_quality()
        combobox_idx = self.top_menu.combo_dataQuality.currentIndex()
        self.dcs[self.current_page].data_quality_idx = combobox_idx

        if combobox_idx == 0:
            # None
            self.dcs[self.current_page].data_quality = None
        elif combobox_idx == 1:
            # Auto
            self.dcs[self.current_page].data_quality = txt_auto_quality
        else:
            # Manual Selection of L1, L2, L3, L4
            self.dcs[self.current_page].data_quality = self.top_menu.combo_dataQuality.currentText()

    def reload_auto_data_quality(self):
        if self.dcs[self.current_page].pixel_end_n is None:
            return
        right = self.dcs[self.current_page].pixel_end_n
        txt_auto_quality = util.get_data_quality(right)
        self.top_menu.combo_dataQuality.setItemText(1, "Auto({})".format(txt_auto_quality))
        return txt_auto_quality


    def sig_binding(self):
        # self.top_menu.open_img_file.triggered.connect(self.menu_open_image_file)
        # self.top_menu.open_img_stack_mrc.triggered.connect(lambda: self.menu_open_image_stack('.mrc'))
        # self.top_menu.open_img_stack_csv.triggered.connect(lambda: self.menu_open_image_stack('.csv'))
        # self.top_menu.open_img_stack_tiff.triggered.connect(lambda: self.menu_open_image_stack('.tiff'))
        # self.top_menu.open_img_stack_png.triggered.connect(lambda: self.menu_open_image_stack('.png'))
        # self.top_menu.open_img_stack_txt.triggered.connect(lambda: self.menu_open_image_stack('.txt'))
        # self.top_menu.open_img_stack_custom.triggered.connect(lambda: self.menu_open_image_stack('.custom'))
        # self.top_menu.open_img_stack_jpg.triggered.connect(lambda: self.menu_open_image_stack('.jpg'))
        # self.top_menu.open_img_stack_jpg.triggered.connect(lambda: self.menu_open_image_stack('.jpeg'))
        # self.top_menu.open_preset.triggered.connect(self.menu_load_preset)
        # self.top_menu.save_preset.triggered.connect(self.menu_save_preset)
        # self.top_menu.open_preset_stack.triggered.connect(self.menu_open_preset_stack)
        # self.top_menu.save_preset_stack.triggered.connect(self.menu_save_presets)
        # self.top_menu.open_azavg_only.triggered.connect(self.menu_open_azavg_only)
        # self.top_menu.open_azavg_stack_csv.triggered.connect(lambda : self.menu_open_azavg_stack("csv"))
        # self.top_menu.open_azavg_stack_txt.triggered.connect(lambda: self.menu_open_azavg_stack("txt"))
        # self.top_menu.open_azavg_stack_azavg_csv.triggered.connect(lambda: self.menu_open_azavg_stack("azavg.csv"))
        # self.top_menu.open_azavg_stack_azavg_txt.triggered.connect(lambda: self.menu_open_azavg_stack("azavg.txt"))
        # self.top_menu.save_azavg_only.triggered.connect(self.menu_save_azavg_only)

        self.top_menu.combo_dataQuality.currentIndexChanged.connect(self.set_data_quality)
        self.top_menu.averaging_gr.triggered.connect(self.menu_util_averaging_gr)
        self.PDF_analyser.graph_Iq_panel.setting.spinBox_range_right.valueChanged.connect(self.set_data_quality)
        self.PDF_analyser.graph_Iq_panel.region.sigRegionChangeFinished.connect(self.set_data_quality)

        self.top_menu.btn_left.clicked.connect(self.btn_page_left_clicked)
        self.top_menu.btn_right.clicked.connect(self.btn_page_right_clicked)

    def menu_util_averaging_gr(self):
        self.averaging_multiple_gr_viewer = averaging_multiple_gr.Viewer(self)
        self.averaging_multiple_gr_viewer.show()
        pass

    def load_dc(self,index):
        # to reduce the memory
        if not len(self.dcs) == 1 and hasattr(self,"current_page")\
                and len(self.dcs) > self.current_page:
            self.dcs[self.current_page].release()
        self.current_page = index

        # update quality number
        self.reload_auto_data_quality()
        if self.dcs[self.current_page].data_quality_idx is not None:
            self.top_menu.combo_dataQuality.setCurrentIndex(self.dcs[self.current_page].data_quality_idx)
        else:
            self.top_menu.combo_dataQuality.setCurrentIndex(0)

        # show image
        self.dcs[self.current_page].image_ready()

        # Update profile_extraction ui
        self.profile_extraction.update_dc(self.dcs[self.current_page])

        # Update pdf_analyser ui
        self.PDF_analyser.put_datacube(self.dcs[self.current_page])

        # Set program title as sample path
        if self.dcs[self.current_page].load_file_path is not None:
            self.setWindowTitle(definitions.PROGRAM_NAME + " : " + self.dcs[self.current_page].load_file_path)

        # Set index label
        self.top_menu.lbl_current_num.setText(str(self.current_page + 1) + "/" + str(len(self.dcs)))

        # mask module
        self.profile_extraction.mask_module.update_img(self.dcs[self.current_page].raw_img)

    def apply_element_to_all(self, datacube):
        for dc in self.dcs:
            dc.ds = datacube.ds
            dc.element_nums = datacube.element_nums
            dc.element_ratio = datacube.element_ratio

    def menu_open_image_file(self):
        load_paths = []
        path,_ = QtWidgets.QFileDialog.getOpenFileNames(self,'open')
        if len(path) == 0:
            return
        load_paths.extend(path)
        self.dcs.clear()
        self.dcs.extend([DataCube(path,'image') for path in load_paths])
        self.load_dc(0)

    def menu_open_image_stack(self, file_type):
        load_paths = []
        path = QtWidgets.QFileDialog.getExistingDirectory(None, 'open')
        if not path:
            return
        load_paths.extend(file.get_file_list_from_path(path, file_type))
        if len(load_paths) == 0:
            QtWidgets.QMessageBox.about(None, "No file found", "No file found")
            return
        dcs = [DataCube(path, 'image') for path in load_paths]
        if dcs is None:
            return
        self.dcs.clear()
        self.dcs.extend(dcs)
        self.load_dc(0)

    def menu_open_preset_stack(self):
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
        self.load_dc(0)

    def menu_open_azavg_only(self, azavg=None):  # azavg arguments is for averaging_multiple_gr.py
        if azavg is None or azavg is False:
            fp, _ = QtWidgets.QFileDialog.getOpenFileName(self, filter="csv (*.csv); text file (*.txt)")
            if fp is '':
                return
            dc = DataCube(file_path=fp,file_type='azavg')
            self.dcs.clear()
            self.dcs.append(dc)
        else:
            self.dcs.clear()
            self.dcs.append(DataCube())
            self.dcs[0].azavg = azavg
        self.load_dc(0)

    def menu_open_azavg_stack(self, ext):  # azavg arguments is for averaging_multiple_gr.py
        dirpth = QtWidgets.QFileDialog.getExistingDirectory(self, '')
        if dirpth is '':
            return
        lst1 = file.get_file_list_from_path(dirpth, ext)

        if len(lst1) == 0:
            QMessageBox.about(self, "", "No file detected")
            return
        dc = [DataCube(file_path=pth, file_type='azavg') for pth in lst1]
        self.dcs.clear()
        self.dcs.extend(dc)
        self.load_dc(0)

    def menu_load_preset(self):
        dc = file.load_preset()
        if not dc:
            return
        self.dcs.clear()
        self.dcs.append(dc)
        self.load_dc(0)

    def menu_save_current_preset(self):
        if len(self.dcs) == 0:
            QMessageBox.about(self,"","No data is loaded")
            return

        if not self.dcs[0].preset_file_path:
            self.menu_save_current_preset_as()
            return

        self.PDF_analyser.manualfit()
        file.save_preset([self.dcs[self.current_page]], self, None, stack=False, saveas=False)

    def menu_save_current_preset_as(self):
        if len(self.dcs) == 0:
            QMessageBox.about(self,"","No data is loaded")
            return

        self.PDF_analyser.manualfit()
        fpth = QtWidgets.QFileDialog.getExistingDirectory(self,"")
        if not fpth:
            return
        file.save_preset([self.dcs[self.current_page]], self, fpth, stack=False, saveas=True)

    def menu_save_all_preset(self):
        if len(self.dcs) == 0:
            QMessageBox.about(self,"","No data is loaded")
            return

        if not self.dcs[0].preset_file_path:
            self.menu_save_all_preset_as()
            return

        temp_page_num = self.current_page
        for i in range(len(self.dcs)):
            self.load_dc(i)
            self.PDF_analyser.manualfit()
            file.save_preset([self.dcs[self.current_page]], self, None, stack=True, saveas=False)
        self.load_dc(temp_page_num)

    def menu_save_all_preset_as(self):
        if len(self.dcs) == 0:
            QMessageBox.about(self,"","No data is loaded")
            return

        fpth = QtWidgets.QFileDialog.getExistingDirectory(self, "")
        if not fpth:
            return

        temp_page_num = self.current_page
        for i in range(len(self.dcs)):
            self.load_dc(i)
            self.PDF_analyser.manualfit()
            file.save_preset([self.dcs[self.current_page]], self, fpth, stack=True, saveas=True)
        self.load_dc(temp_page_num)

    def menu_save_azavg_only(self):
        if self.dcs[self.current_page].azavg is not None:
            file.save_azavg_only(self.dcs[self.current_page].azavg)

    def menu_save_azavg_stack(self):
        file.save_azavg_stack(self.dcs)

    def btn_page_left_clicked(self):
        if hasattr(self, "current_page") and not self.current_page == 0:
            self.load_dc(self.current_page - 1)

    def btn_page_right_clicked(self):
        if hasattr(self, "current_page") and not self.current_page == len(self.dcs) - 1:
            self.load_dc(self.current_page + 1)

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        if e.key() in [QtCore.Qt.Key.Key_PageUp] or \
                ((e.modifiers() & QtCore.Qt.Modifier.CTRL) and e.key() == QtCore.Qt.Key.Key_Left):
            self.btn_page_left_clicked()
        if e.key() in [QtCore.Qt.Key.Key_PageDown] or \
                ((e.modifiers() & QtCore.Qt.Modifier.CTRL) and e.key() == QtCore.Qt.Key.Key_Right):
            self.btn_page_right_clicked()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        util.default_setting.intensity_range_1 = self.profile_extraction.control_panel.settingPanel.spinBox_irange1.value()
        util.default_setting.intensity_range_2 = self.profile_extraction.control_panel.settingPanel.spinBox_irange2.value()
        util.default_setting.slice_count = self.profile_extraction.control_panel.settingPanel.spinBox_slice_count.value()
        util.default_setting.show_center_line = self.profile_extraction.control_panel.settingPanel.chkBox_show_centerLine.isChecked()
        util.default_setting.save_settings()
        super().closeEvent(a0)


if __name__ == '__main__':
    qtapp = QtWidgets.QApplication.instance()
    if not qtapp:
        qtapp = QtWidgets.QApplication(sys.argv)
    app = DataViewer()
    app.show()
    sys.exit(qtapp.exec_())
