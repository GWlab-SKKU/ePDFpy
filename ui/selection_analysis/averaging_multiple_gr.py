from PyQt5 import QtWidgets, QtCore
import ui.main as main
import pyqtgraph as pg
import os
import glob
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt, QRectF
import numpy as np
from ui import ui_util
import datacube
from PyQt5.QtGui import QColor
import util
from ui.selection_analysis.column_selector import ColumnSelector

pg.setConfigOptions(antialias=True)

class DataViewer(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        viewer = Viewer(self)
        self.setCentralWidget(viewer)

class Viewer(QtWidgets.QWidget):
    str_parameter = r"\Parameters.csv"
    str_data_r = r"\Data_r.csv"
    str_data_q = r"\Data_q.csv"

    def __init__(self, mainWindow, profile_extraction=None):
        super().__init__()

        self.grCubes = []
        self.avg_azavg = None
        self.profile_extraction = profile_extraction
        self.mainWindow = mainWindow
        self.data_x = None
        self.data_y = None

        self.initui()

        self.binding()

    def initui(self):
        self.layout = QtWidgets.QVBoxLayout()

        self.setLayout(self.layout)
        self.leftPanel = LeftPanel()
        self.rightPanel = GraphPanel()
        self.average_plot = self.rightPanel.graphView.plot(pen=pg.mkPen(255,255,255,width=5))
        self.variance_plot = self.rightPanel.graphView.plot(pen=pg.mkPen(255, 255, 255, width=5))

        self.splitter_horizontal = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter_horizontal.addWidget(self.leftPanel)
        self.splitter_horizontal.addWidget(self.rightPanel)


        self.menu_bar = self.create_menu_bar()

        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.menu_bar)
        self.layout.addWidget(self.splitter_horizontal)

    def binding(self):
        self.menu_open_gr.triggered.connect(self.open_btn_clicked)
        self.menu_open_csv.triggered.connect(self.open_csv_clicked)
        self.leftPanel.graph_x_data_select_area.cmb_x_data.currentTextChanged.connect(self.x_axis_changed)
        self.leftPanel.graph_y_data_select_area.radio_grp.buttonToggled.connect(self.y_axis_changed)
        self.leftPanel.graph_range_area.spinbox_range_min.valueChanged.connect(self.update_graph)
        self.leftPanel.graph_range_area.spinbox_range_max.valueChanged.connect(self.update_graph)
        self.leftPanel.graph_ops_select_area.chkbox_average.clicked.connect(self.update_graph)
        self.leftPanel.graph_ops_select_area.chkbox_variance.clicked.connect(self.update_graph)
        self.menu_open_txt.triggered.connect(self.open_txt_clicked)
        self.menu_open_custom.triggered.connect(self.open_custom_clicked)
        # self.leftPanel.btn_group.btn_save_gr_avg.clicked.connect(self.save_gr_avg)
        # self.leftPanel.btn_group.btn_save_intensity_avg.clicked.connect(self.save_intensity_avg)
        # self.leftPanel.btn_group.btn_open_analyzer.clicked.connect(self.open_analyzer)

    def save_gr_avg(self):
        # calculate avg
        fp,ext = QtWidgets.QFileDialog.getSaveFileName(self, filter="CSV Files (*.csv);;All Files (*)")
        if fp == "":
            return
        df = pd.DataFrame({'r':self.grCubes[0].r,'Gr':self.avg})
        df.to_csv(fp+".csv",index=False)

    def save_intensity_avg(self):
        fp, ext = QtWidgets.QFileDialog.getSaveFileName(self, filter="text file (*.txt);;All Files (*)")
        if fp == "":
            return
        self.averaging_profile_intensity()
        np.savetxt(fp+".txt", self.avg_azavg)

    def open_analyzer(self):
        if self.grCubes is None or len(self.grCubes) is 0:
            pass
        else:
            self.averaging_profile_intensity()

        if self.avg_azavg is not None:
            self.profile_extraction.menu_open_azavg_only(self.avg_azavg)
            print(np.sum(self.avg_azavg))

    def averaging_profile_intensity(self):
        azavg_list = []
        shortest_end = 1000000
        for grCube in self.grCubes:
            if grCube.chkbox_module.chkbox.isChecked():
                azavg = np.loadtxt(grCube.azavg_file_path)
                if shortest_end > len(azavg):
                    shortest_end = len(azavg)
                azavg_list.append(azavg)
        azavg_list = [azavg[:shortest_end] for azavg in azavg_list]
        avg_azavg = np.average(np.array(azavg_list), axis=0).transpose()
        self.avg_azavg = avg_azavg

    def verify_files(self, dcs):
        # column header corresponding
        if not util.xor([dc.original_data_has_column for dc in dcs]):
            QMessageBox.about(self, "Error1", "column header doesn't match each other")
            return False
        # search intersection
        lst_set = [set(dc.pd_data.columns) for dc in dcs]
        if len(set.intersection(*lst_set)) == 0:
            QMessageBox.about(self, "Error2", "column header doesn't match each other. \n"
                                                    "Detected columns:{}"
                                    .format(set.union(*lst_set)-set.intersection(*lst_set)))
            return False
        if len(set.union(*lst_set)) != len(set.intersection(*lst_set)):
            reply = QMessageBox.Information(self, "Select", "Only shared headers will show. \n"
                                                    "Shared columns:{} \n"
                                                    "Different columns:{}"
                                    .format(set.intersection(*lst_set),set.symmetric_difference(*lst_set)))
            if reply == QMessageBox.No:
                return False
        return True

    def open_file_clicked(self):
        fp, _ = QtWidgets.QFileDialog.getOpenFileName()
        if ".preset.csv" in fp:
            dc = datacube.DataCube(fp,file_type="preset")
        else:
            dc = datacube.DataCube(fp)
        return dc

    def open_csv_clicked(self):
        #### get files ####
        dcs = self.get_dcs_from_folder("*.csv")
        if dcs is None:
            return
        self.open_stack(dcs)

    def open_txt_clicked(self):
        #### get files ####
        dcs = self.get_dcs_from_folder("*.txt")
        if dcs is None:
            return
        self.open_stack(dcs)

    def open_custom_clicked(self):
        #### get files ####
        text, _ = QtWidgets.QInputDialog.getText(None, "Type custom name", "e.g) *.csv, *azavg*.txt")
        dcs = self.get_dcs_from_folder(text)
        if dcs is None:
            return
        self.open_stack(dcs)



    def open_stack(self, dcs):
        if len(dcs) == 0:
            QMessageBox.about(self,"Error","No data detected on the folder")

        #### verify files ####
        if not self.verify_files(dcs):
            return

        #### grcube load ####
        [grCube.clear() for grCube in self.grCubes]
        self.grCubes.clear()
        self.grCubes.extend(dcs)

        # cut data
        self.data_cut(dcs)

        # set x axis
        columns = dcs[0].pd_data.columns.to_list()
        combo_lst = ["None"]
        combo_lst.extend(columns)
        self.leftPanel.graph_x_data_select_area.cmb_x_data.blockSignals(True)
        self.leftPanel.graph_x_data_select_area.cmb_x_data.clear()
        self.leftPanel.graph_x_data_select_area.cmb_x_data.addItems(combo_lst)
        self.leftPanel.graph_x_data_select_area.cmb_x_data.blockSignals(False)

        # set y axis
        self.leftPanel.graph_y_data_select_area.clear_radio()
        for clmn in columns:
            radio = self.leftPanel.graph_y_data_select_area.add_radio(clmn)
            radio.toggled.connect(self.y_axis_changed)
        self.leftPanel.graph_y_data_select_area.radio_grp.blockSignals(True)
        self.leftPanel.graph_y_data_select_area.radio_grp.buttons()[0].setChecked(True)
        self.leftPanel.graph_y_data_select_area.radio_grp.blockSignals(False)

        self.change_range()
        self.set_data()

        for idx, grCube in enumerate(self.grCubes):
            grCube.color = pg.intColor(idx, minValue=200, alpha=200)
            color = grCube.color
            grCube.plotItem = ui_util.HoverableCurveItem(grCube.data_x, grCube.data_y, pen=pg.mkPen(color), setAcceptHoverEvent=True)
            self.rightPanel.graphView.addItem(grCube.plotItem)
            grCube.chkbox_module = self.leftPanel.graph_list_area.add_module(grCube.load_file_path, color)
            grCube.chkbox_module.chkbox.toggled.connect(grCube.plot_show_hide)
            grCube.chkbox_module.chkbox.toggled.connect(self.calculate_average)
            grCube.chkbox_module.chkbox.toggled.connect(self.calculate_variance)
            grCube.binding_event()
        # columns_to_add = columns
        # if x_axis in columns_to_add:
        #     columns_to_add.remove(x_axis)
        # self.leftPanel.graph_column_select_area.add_radio()
        #### draw graphs ####
        # for dc in dcs:
        #     self.draw_graphs(dc)

        if self.average_plot is not None:
            if self.leftPanel.graph_ops_select_area.chkbox_average.isChecked():
                self.calculate_average()
            else:
                self.average_plot.hide()
            if self.leftPanel.graph_ops_select_area.chkbox_variance.isChecked():
                self.calculate_variance()
            else:
                self.variance_plot.hide()


    def x_axis_changed(self):
        current_x_axis = self.leftPanel.graph_x_data_select_area.cmb_x_data.currentText()
        buttons = self.leftPanel.graph_y_data_select_area.radio_grp.buttons()
        for idx, radio in enumerate(buttons):
            if current_x_axis == radio.text():
                radio.setEnabled(False)
                if radio.isChecked():
                    buttons[np.mod(idx+1,len(buttons))].setChecked(True)
            else:
                radio.setEnabled(True)
        self.change_range()
        self.set_data()
        self.update_graph()

    def y_axis_changed(self):
        self.set_data()
        self.update_graph()

    def update_graph(self):
        l = self.leftPanel.graph_range_area.spinbox_range_min.value()
        r = self.leftPanel.graph_range_area.spinbox_range_max.value()

        if len(self.grCubes) == 0:
            return
        if not hasattr(self.grCubes[0], 'data_x'):
            return
        if self.grCubes[0].plotItem is None:
            return

        idx_l, value_l = util.find_nearest(self.grCubes[0].data_x, l)
        idx_r, value_r = util.find_nearest(self.grCubes[0].data_x, r)

        for grCube in self.grCubes:
            grCube.plotItem.setData(grCube.data_x[idx_l:idx_r],grCube.data_y[idx_l:idx_r])

        # average
        if self.leftPanel.graph_ops_select_area.chkbox_average.isChecked():
            self.average_plot.setVisible(True)
            self.calculate_average()
        else:
            self.average_plot.setVisible(False)
        # std
        if self.leftPanel.graph_ops_select_area.chkbox_variance.isChecked():
            self.variance_plot.setVisible(True)
            self.calculate_variance()
        else:
            self.variance_plot.setVisible(False)

        self.rightPanel.graphView.autoRange()

    def data_cut(self, cubes):
        min_length = min([len(grCube.pd_data) for grCube in cubes])
        for grCube in cubes:
            grCube.pd_data = grCube.pd_data[0:min_length]

    def change_range(self):
        x_axis = self.leftPanel.graph_x_data_select_area.cmb_x_data.currentText()
        if x_axis == "None":
            l = 0
            r = len(self.grCubes[0].pd_data)-1
        else:
            nparr = self.grCubes[0].pd_data[x_axis].to_numpy()
            l = nparr.min()
            r = nparr.max()
        self.leftPanel.graph_range_area.spinbox_range_min.blockSignals(True)
        self.leftPanel.graph_range_area.spinbox_range_max.blockSignals(True)
        self.leftPanel.graph_range_area.spinbox_range_min.setMinimum(l)
        self.leftPanel.graph_range_area.spinbox_range_max.setMinimum(l)
        self.leftPanel.graph_range_area.spinbox_range_min.setMaximum(r)
        self.leftPanel.graph_range_area.spinbox_range_max.setMaximum(r)

        self.leftPanel.graph_range_area.spinbox_range_min.setValue(l)
        self.leftPanel.graph_range_area.spinbox_range_max.setValue(r)
        self.leftPanel.graph_range_area.spinbox_range_min.blockSignals(False)
        self.leftPanel.graph_range_area.spinbox_range_max.blockSignals(False)
        self.data_range_l = l
        self.data_range_r = r

    def set_data(self):
        button = self.leftPanel.graph_y_data_select_area.radio_grp.checkedButton()
        if button is None:
            return
        for grcube in self.grCubes:
            str_xaxis = self.leftPanel.graph_x_data_select_area.cmb_x_data.currentText()
            if str_xaxis == "None":
                grcube.data_x = np.arange(0,len(grcube.pd_data))
            else:
                grcube.data_x = grcube.pd_data[str_xaxis].to_numpy()
            str_yaxis = button.text()
            grcube.data_y = grcube.pd_data[str_yaxis].to_numpy()


    def draw_graphs(self, dcs:[datacube.DataCube]):
        for idx, dc in enumerate(dcs):
            dc:datacube.DataCube
            color = pg.intColor(idx, minValue=200)
            graph_name = os.path.split(dc.load_file_path)[1]
            if len(dc.original_data) == 1:
                plot_ = self.rightPanel.graphView.plot(dc.original_data, name=graph_name, pen=pg.mkPen(color))
            grCube.plotItem = plot_
        pass

    def open_stack_txt_clicked(self):

        pass

    def open_btn_clicked(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self,'open')
        self.setWindowTitle(path)
        if path == "":
            return

        for grCube in self.grCubes: grCube.clear()
        self.grCubes.clear()

        data_r_files, data_azav_files = self.get_gr_files(path)
        if len(data_r_files) == 0 :
            QMessageBox.about(self,"Error","No data detected on the folder")

        ################## MAIN ####################
        for idx, data_r_file in enumerate(data_r_files):
            grCube = GrCube()
            grCube.unit_path = data_r_file

            # graph name
            # graph_name = data_r_file[data_r_file.rfind("_")+1:data_r_file.find("Data_r.csv")]
            graph_name = data_r_file[data_r_file.rfind("_") + 1:data_r_file.find("r30.csv")]

            # put parameters

            # load and plot
            grCube.r, grCube.Gr = self.load_Gr_file(data_r_file)
            grCube.color = pg.intColor(idx,minValue=200)
            grCube.r_file_path = data_r_file
            grCube.azavg_file_path = data_azav_files[idx]
            color = grCube.color
            plot_ = self.rightPanel.graphView.plot(grCube.r,grCube.Gr,name=graph_name,pen=pg.mkPen(color))
            grCube.plotItem = plot_
            grCube.chkbox_module = self.leftPanel.graph_list_area.add_module(os.path.split(data_r_file)[1],color)
            grCube.chkbox_module.chkbox.clicked.connect(grCube.plot_show_hide)
            grCube.chkbox_module.chkbox.clicked.connect(self.calculate_average)

            grCube.binding_event()

            self.grCubes.append(grCube)

        self.calculate_average()
        self.rightPanel.graphView.autoRange()
        self.rightPanel.graphView.setRange(xRange=[0,10])

    def load_Gr_file(self, path):
        df = pd.read_csv(path)
        r = df['r'].to_numpy()
        Gr = df['Gr'].to_numpy()
        return r, Gr

    def get_dcs_from_folder(self, file_extension, folder=None):
        if folder == None:
            folder = QtWidgets.QFileDialog.getExistingDirectory()
        if folder == "":
            return
        files = Path(folder).rglob(file_extension)
        dcs = [GrCube(file_path=str(file.absolute())) for file in files]
        return dcs

    def get_gr_files(self, folder):
        # Old data search #
        csvfiles = Path(folder).rglob("*_r30_*.csv")
        gr_path_list = []
        azavg_path_lst = []
        for file in csvfiles:
            # if file.name in ["diagonal.csv", "diagonal_1.csv", "line.csv"]:
            #     continue
            # if "r30" not in file.name:
            #     continue

            fp1 = str(file.absolute())
            fp1 = os.path.split(fp1)[1]
            fp1 = fp1[:fp1.rfind("_r30_")]
            search_name = fp1[:32]
            rglob = Path(folder).rglob(search_name+"*azav*.txt")
            searched_files = []
            for fp2 in rglob:
                searched_files.append(str(fp2.absolute()))

            if len(searched_files) > 0:
                # print(searched_files[0])
                gr_path_list.append(str(file.absolute()))
                azavg_path_lst.append(searched_files[0])
            else:
                QMessageBox.about(self,"Not Found","failed to find azav file for {}".format(file))
        if len(azavg_path_lst)>0:
            QMessageBox.about(self, "Not Found", "Only file name with r30 files are loaded")
            return gr_path_list, azavg_path_lst

        # Old data search2 #
        csvfiles = Path(folder).rglob("*Data_r.csv")
        for file in csvfiles:
            fp1 = str(file.absolute())
            fp1 = os.path.split(fp1)[1]
            fp1 = fp1[:fp1.rfind("Data_r.csv")-1]
            search_name = fp1[:32]
            rglob = Path(folder).rglob(search_name+"*azav*.txt")
            searched_files = []
            for fp2 in rglob:
                searched_files.append(str(fp2.absolute()))

            if len(searched_files) > 0:
                # print(searched_files[0])
                gr_path_list.append(str(file.absolute()))
                azavg_path_lst.append(searched_files[0])
            else:
                QMessageBox.about(self,"Not Found","failed to find azav file for {}".format(file))


        # New data search #
        csvfiles = Path(folder).rglob("*.r.csv")
        for file in csvfiles:
            fp1 = str(file.absolute())
            fp1 = os.path.split(fp1)[1]
            fp1 = fp1[:fp1.rfind(".r.csv")]
            print(fp1)
            rglob = Path(folder).rglob(fp1 + ".azavg.txt")
            searched_files = []
            for fp2 in rglob:
                searched_files.append(str(fp2.absolute()))

            if len(searched_files) > 0:
                # print(searched_files[0])
                gr_path_list.append(str(file.absolute()))
                azavg_path_lst.append(searched_files[0])
            else:
                QMessageBox.about(self,"Not Found","failed to find azav file for {}".format(file))

        return gr_path_list, azavg_path_lst

    def calculate_average(self):
        avg_lst = [grCube.data_y for grCube in self.grCubes if grCube.plotItem.isVisible()]
        self.avg = np.average(np.array(avg_lst).transpose(), axis=1)
        self.average_plot.setData(self.grCubes[0].data_x, self.avg)

    def calculate_variance(self):
        avg_lst = [grCube.data_y for grCube in self.grCubes if grCube.plotItem.isVisible()]
        self.var = np.std(np.array(avg_lst).transpose(), axis=1)
        self.variance_plot.setData(self.grCubes[0].data_x, self.var)

    # def hovering_event(self):
    #     for grCube in self.grCubes:
    #         grCube.chkbox_module:GraphModule
    #         grCube.chkbox_module.enterEvent_list.append(grCube.plotItem)

    def create_menu_bar(self):
        self.mainWindow = self.mainWindow
        self.menubar = self.mainWindow.menuBar()

        self.menu_open_csv = QtWidgets.QAction("csv", self.mainWindow)
        self.menu_open_txt = QtWidgets.QAction("txt", self.mainWindow)
        self.menu_open_preset = QtWidgets.QAction("preset", self.mainWindow)
        self.menu_open_gr = QtWidgets.QAction("gr", self.mainWindow)
        self.menu_open_custom = QtWidgets.QAction("Custom Name", self.mainWindow)

        self.menubar: QtWidgets.QMenuBar
        self.menubar.setMaximumHeight(25)
        self.menubar.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum,
                                                         QtWidgets.QSizePolicy.Policy.Minimum))

        open_menu = self.menubar.addMenu("     &Open     ")
        open_menu.addAction(self.menu_open_csv)
        open_menu.addAction(self.menu_open_txt)
        open_menu.addAction(self.menu_open_preset)
        open_menu.addAction(self.menu_open_gr)
        open_menu.addAction(self.menu_open_custom)
        open_menu.addSeparator()

        save_menu = self.menubar.addMenu("     &Save     ")
        return self.menubar


class GrCube(datacube.DataCube):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_file_path = None
        self.color: QColor
        self.color = None
        self.plotItem = None
        self.chkbox_module:GraphModule
        self.chkbox_module = None

    def plot_show_hide(self):
        if self.chkbox_module.chkbox.isChecked():
            self.plotItem.show()
        else:
            self.plotItem.hide()

    def plot_hide(self):
        self.chkbox_module.chkbox.setChecked(False)
        self.plotItem.hide()

    def clear(self):
        self.plotItem.clear()
        self.chkbox_module.deleteLater()
        self.chkbox_module = None

    def binding_event(self):
        # widget enter event
        enter_pen = pg.mkPen(self.color,width=5)
        default_pen = pg.mkPen(self.color,width=1)
        print(enter_pen)
        self.chkbox_module.enterEvent_list.append(lambda: self.plotItem.setPen(enter_pen))
        self.chkbox_module.leaveEvent_list.append(lambda: self.plotItem.setPen(default_pen))
        self.plotItem.sigCurveHovered.connect(lambda: self.plotItem.setPen(enter_pen))
        self.plotItem.sigCurveNotHovered.connect(lambda: self.plotItem.setPen(default_pen))

        self.plotItem.sigCurveHovered.connect(lambda: self.plotItem.setPen(enter_pen))
        self.plotItem.sigCurveNotHovered.connect(lambda: self.plotItem.setPen(default_pen))
        self.plotItem.sigCurveClicked.connect(self.plot_hide)


class GraphPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.graphView = ui_util.CoordinatesPlotWidget(title='G(r)', setYScaling=False, button1mode=True)
        self.axis1 = pg.InfiniteLine(angle=0)
        self.graphView.addItem(self.axis1)
        self.layout.addWidget(self.graphView)


class LeftPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.splitter_left_vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.layout)

        self.graph_list_area = self.GraphListArea()
        self.graph_ops_select_area = self.GraphOpsArea()
        self.graph_y_data_select_area = self.GraphYDataSelectArea()
        self.graph_range_area = self.GraphRangeArea()
        self.graph_x_data_select_area = self.GraphXDataSelectArea()
        # self.btn_group = self.ButtonListWidget()

        self.splitter_left_vertical.addWidget(self.graph_list_area)
        self.splitter_left_vertical.addWidget(self.graph_ops_select_area)
        self.splitter_left_vertical.addWidget(self.graph_x_data_select_area)
        self.splitter_left_vertical.addWidget(self.graph_y_data_select_area)
        self.splitter_left_vertical.addWidget(self.graph_range_area)

        self.splitter_left_vertical.setStretchFactor(0, 5)
        # self.splitter_left_vertical.setStretchFactor(1, 5)
        # self.splitter_left_vertical.setStretchFactor(2, 5)
        # self.splitter_left_vertical.addWidget(self.btn_group)

        self.graph_y_data_select_area.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                                         QtWidgets.QSizePolicy.Policy.Maximum))

        self.layout.addWidget(self.splitter_left_vertical)

    class GraphXDataSelectArea(QtWidgets.QGroupBox):
        def __init__(self):
            super().__init__("X data")
            self.layout = QtWidgets.QVBoxLayout()
            self.setLayout(self.layout)
            self.cmb_x_data = QtWidgets.QComboBox()
            self.layout.addWidget(self.cmb_x_data)

    class GraphRangeArea(QtWidgets.QGroupBox):
        def __init__(self):
            super().__init__("Show range")
            self.layout = QtWidgets.QVBoxLayout()
            self.setLayout(self.layout)
            self.spinbox_range_min = QtWidgets.QDoubleSpinBox()
            self.spinbox_range_max = QtWidgets.QDoubleSpinBox()
            self.layout.addWidget(self.spinbox_range_min)
            self.layout.addWidget(self.spinbox_range_max)

    class GraphOpsArea(QtWidgets.QGroupBox):
        def __init__(self):
            super().__init__("Operation")
            self.layout = QtWidgets.QVBoxLayout()
            self.setLayout(self.layout)
            self.chkbox_average = QtWidgets.QCheckBox("Average")
            self.chkbox_variance = QtWidgets.QCheckBox("Variance")
            self.layout.addWidget(self.chkbox_average)
            self.layout.addWidget(self.chkbox_variance)

    class GraphYDataSelectArea(QtWidgets.QGroupBox):
        def __init__(self):
            super().__init__("Y data")
            self.layout = QtWidgets.QVBoxLayout()
            self.setLayout(self.layout)
            self.radio_grp = QtWidgets.QButtonGroup()

        def add_radio(self, name):
            radio = QtWidgets.QRadioButton(name)
            self.radio_grp.addButton(radio)
            self.layout.addWidget(radio)
            return radio

        def clear_radio(self):
            for radio in self.radio_grp.buttons():
                self.radio_grp.removeButton(radio)
                self.layout.removeWidget(radio)
                radio.deleteLater()

    class GraphListArea(QtWidgets.QScrollArea):
        def __init__(self):
            super().__init__()
            self.graph_group_widget = QtWidgets.QWidget()
            self.graph_group_widget.layout = QtWidgets.QVBoxLayout()
            self.graph_group_widget.setLayout(self.graph_group_widget.layout)
            self.graph_group_widget.layout.addSpacing(0)
            self.graph_group_widget.layout.addStretch(0)
            self.graph_group_widget.layout.setSpacing(5)
            self.setWidget(self.graph_group_widget)

            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.setWidgetResizable(True)

        def add_module(self, text, color):
            print("module:",text)
            module = GraphModule()
            module.set_text(text)
            module.set_color(color)
            self.graph_group_widget.layout.insertWidget(0, module)
            return module

    class ButtonListWidget(QtWidgets.QWidget):
        def __init__(self):
            super().__init__()
            self.btn_open_folder = QtWidgets.QPushButton("Open Folder")
            self.btn_save_gr_avg = QtWidgets.QPushButton("save G(r) avg")
            self.btn_save_intensity_avg = QtWidgets.QPushButton("save selected intensity avg")
            self.btn_open_analyzer = QtWidgets.QPushButton("Open analyzer")

            self.layout = QtWidgets.QVBoxLayout()
            self.layout.setContentsMargins(0, 0, 0, 0)
            self.setLayout(self.layout)

            self.layout.addWidget(self.btn_open_folder)
            self.layout.addWidget(self.btn_save_gr_avg)
            self.layout.addWidget(self.btn_save_intensity_avg)
            self.layout.addWidget(self.btn_open_analyzer)


class GraphModule(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.enterEvent_list = []
        self.leaveEvent_list = []
        self.enterEvent_list = [self.enter_color]
        self.leaveEvent_list = [self.exit_color]
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.setLayout(layout)
        self.chkbox = QtWidgets.QCheckBox("")
        self.chkbox.setChecked(True)
        self.textbox = QtWidgets.QPlainTextEdit("")
        self.textbox.setReadOnly(True)
        self.textbox.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.color = None
        self.color_txt = None
        self.dark_color_txt = None
        self.bright_color_txt = None
        self.textbox.setMaximumHeight(20)
        layout.addWidget(self.chkbox)

        # layout.addWidget(self.textbox)

    # def set_text(self,text):
    #     self.textbox.setPlainText(text)

    def set_text(self,text):
        self.chkbox.setText(text)

    def set_color(self, color):
        self.color = color
        self.dark_color = self.color.lighter()
        self.color_txt = "rgba({}, {}, {}, {});".format(color.red(), color.green(), color.blue(), color.alpha())
        self.dark_color_txt = "rgba({}, {}, {}, {});".format(color.red(), color.green(), color.blue(), 100)
        self.bright_color_txt = "rgba({}, {}, {}, {});".format(color.red(), color.green(), color.blue(), 100)
        self.setStyleSheet("background-color: {};"
                           "padding-top: 10px;"
                           "padding-bottom: 10px;".format(self.color_txt))
        # self.chkbox.styleSheet()
        pass

    def enter_color(self):
        self.setStyleSheet("background-color: {};"
                           "padding-top: 10px;"
                           "padding-bottom: 10px;".format(self.bright_color_txt))

    def exit_color(self):
        self.setStyleSheet("background-color: {};"
                           "padding-top: 10px;"
                           "padding-bottom: 10px;".format(self.color_txt))

    def enterEvent(self, a0: QtCore.QEvent) -> None:
        super().enterEvent(a0)
        for func in self.enterEvent_list:
            func()

    def leaveEvent(self, a0: QtCore.QEvent) -> None:
        super().leaveEvent(a0)
        for func in self.leaveEvent_list:
            func()




if __name__ == "__main__":
    qtapp = QtWidgets.QApplication([])
    # QtWidgets.QMainWindow().show()
    viewer = DataViewer()
    viewer.show()
    qtapp.exec()