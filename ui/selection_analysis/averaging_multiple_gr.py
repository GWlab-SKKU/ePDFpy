from PyQt5 import QtWidgets, QtCore
from ui import main
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
from typing import List
import platform
from ui.selection_analysis.column_selector import ColumnSelector
if platform.system() == 'Darwin':
    default_pen_thickness = 2
    highlight_pen_thickness = 7
else:
    default_pen_thickness = 1
    highlight_pen_thickness = 4
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
        self.grCubes: List[GrCube] = []
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
        self.average_plot = self.rightPanel.graphView.plot(pen=pg.mkPen(255, 255, 255, width=highlight_pen_thickness))
        self.std_plot = self.rightPanel.graphView.plot(pen=pg.mkPen(255, 255, 255, width=highlight_pen_thickness))

        self.splitter_horizontal = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter_horizontal.addWidget(self.leftPanel)
        self.splitter_horizontal.addWidget(self.rightPanel)


        self.menu_bar = self.create_menu_bar()

        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.menu_bar)
        self.layout.addWidget(self.splitter_horizontal)

        self.setStyleSheet(ui_util.get_style_sheet())

    def binding(self):
        self.menu_open_gr.triggered.connect(self.open_gr_clicked)
        self.menu_open_csv.triggered.connect(self.open_csv_clicked)
        self.menu_open_txt.triggered.connect(self.open_txt_clicked)
        self.menu_open_custom.triggered.connect(self.open_custom_clicked)
        self.menu_save_current.triggered.connect(self.save_current_graphs)
        self.menu_save_intensity_profile_avg.triggered.connect(self.save_intensity_avg)
        self.leftPanel.graph_x_data_select_area.cmb_x_data.currentTextChanged.connect(self.x_axis_changed)
        self.leftPanel.graph_range_area.spinbox_range_min.valueChanged.connect(self.update_graph)
        self.leftPanel.graph_range_area.spinbox_range_max.valueChanged.connect(self.update_graph)
        self.leftPanel.graph_range_area.chkbox_legend.toggled.connect(self.update_legend)
        self.leftPanel.graph_ops_select_area.chkbox_average.clicked.connect(self.update_graph)
        self.leftPanel.graph_ops_select_area.chkbox_std.clicked.connect(self.update_graph)

        # self.leftPanel.btn_group.btn_save_gr_avg.clicked.connect(self.save_gr_avg)
        # self.leftPanel.btn_group.btn_save_intensity_avg.clicked.connect(self.save_intensity_avg)
        # self.leftPanel.btn_group.btn_open_analyzer.clicked.connect(self.open_analyzer)
    def update_legend(self):
        checked = self.leftPanel.graph_range_area.chkbox_legend.isChecked()
        if checked:
            self.rightPanel.legend.setVisible(True)
        else:
            self.rightPanel.legend.setVisible(False)

    def save_current_graphs(self):
        plot_to_save = {}

        if len(self.grCubes) == 0:
            return

        # x data
        str_xaxis = self.leftPanel.graph_x_data_select_area.cmb_x_data.currentText()
        if str_xaxis == "None":
            x = np.arange(len(self.grCubes[0].pd_data))
        else:
            x = self.grCubes[0].pd_data[str_xaxis]
        plot_to_save.update({"x":x})

        # y data
        str_yaxis = self.leftPanel.graph_y_data_select_area.radio_grp.checkedButton().text()
        for grCube in self.grCubes:
            if grCube.chkbox_module.isChecked() == True:
                y = grCube.pd_data[str_yaxis]
                plot_to_save.update({grCube.load_file_path:y})

        # average
        if self.leftPanel.graph_ops_select_area.chkbox_average.isChecked():
            avg_lst = [grCube.pd_data[str_yaxis] for grCube in self.grCubes if grCube.plotItem.isVisible()]
            avg = np.average(np.array(avg_lst).transpose(), axis=1)
            plot_to_save.update({"average":avg})

        if self.leftPanel.graph_ops_select_area.chkbox_std.isChecked():
            std_lst = [grCube.pd_data[str_yaxis] for grCube in self.grCubes if grCube.plotItem.isVisible()]
            std_ = np.std(np.array(std_lst).transpose(), axis=1)
            plot_to_save.update({"std": std_})

        df = pd.DataFrame(plot_to_save)
        cols = df.columns.to_list()
        if "std" in cols:
            cols.remove("std")
            cols.insert(1,"std")
        if "average" in cols:
            cols.remove("average")
            cols.insert(1,"average")
        df.columns = cols

        fp, ext = QtWidgets.QFileDialog.getSaveFileName(self, filter="CSV Files (*.csv)")
        if fp == '':
            return
        df.to_csv(fp, index=None)

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
            if grCube.chkbox_module.isChecked():
                azavg = np.loadtxt(grCube.azavg_file_path)
                shortest_end = min(shortest_end,len(azavg))
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
                                    .format(set.union(*lst_set)))
            return False
        if len(set.union(*lst_set)) != len(set.intersection(*lst_set)):
            reply = QMessageBox.information(self, "Select", "Only shared headers will show. \n"
                                                    "Shared columns:{} \n"
                                                    "Different columns:{}"
                                    .format(set.intersection(*lst_set),set.union(*lst_set)-set.intersection(*lst_set)))
            if reply == QMessageBox.No:
                return False
        return True

    def open_csv_clicked(self):
        #### get files ####
        dcs = self.get_dcs_from_folder("*.csv")
        if dcs is None:
            return
        self.menu_save_intensity_profile_avg.setEnabled(False)
        self.open_stack(dcs)
        self.update_graph()
        self.rightPanel.graphView.autoRange()

    def open_txt_clicked(self):
        #### get files ####
        dcs = self.get_dcs_from_folder("*.txt")
        if dcs is None:
            return
        self.menu_save_intensity_profile_avg.setEnabled(False)
        self.open_stack(dcs)
        self.update_graph()
        self.rightPanel.graphView.autoRange()

    def open_custom_clicked(self):
        #### get files ####
        text, status = QtWidgets.QInputDialog.getText(None, "custom name", "Type filtering name using wildcard \ne.g) *Data_q.csv, *azavg*.txt")
        if text == "" or status is False:
            return
        dcs = self.get_dcs_from_folder(text)
        if dcs is None:
            return
        self.menu_save_intensity_profile_avg.setEnabled(False)
        self.open_stack(dcs)
        self.update_graph()
        self.rightPanel.graphView.autoRange()



    def open_stack(self, dcs):
        if len(dcs) == 0:
            QMessageBox.about(self,"Error","No data detected on the folder")

        #### verify files ####
        if not self.verify_files(dcs):
            return

        #### grcube load ####
        for grCube in self.grCubes:
            grCube.clear()
        self.rightPanel.legend.clear()
        self.grCubes.clear()
        self.grCubes.extend(dcs)

        # # cut data
        # self.data_cut(dcs)

        # set x axis
        lst_set = [set(dc.pd_data.columns) for dc in dcs]
        shared_columns = list(set.intersection(*lst_set))
        combo_lst = ["None"]
        combo_lst.extend(shared_columns)
        self.leftPanel.graph_x_data_select_area.cmb_x_data.blockSignals(True)
        self.leftPanel.graph_x_data_select_area.cmb_x_data.clear()
        self.leftPanel.graph_x_data_select_area.cmb_x_data.addItems(combo_lst)
        self.leftPanel.graph_x_data_select_area.cmb_x_data.blockSignals(False)

        # set y axis
        self.leftPanel.graph_y_data_select_area.clear_radio()
        for clmn in shared_columns:
            radio = self.leftPanel.graph_y_data_select_area.add_radio(clmn)
            radio.toggled.connect(self.y_axis_changed)
        radio_grp = self.leftPanel.graph_y_data_select_area.radio_grp
        radio_lst = radio_grp.buttons()
        radio_grp.blockSignals(True)
        radio_lst[0].setChecked(True)
        radio_grp.blockSignals(False)

        self.change_range()
        self.set_data()

        for idx, grCube in enumerate(self.grCubes):
            title = os.path.splitext(os.path.split(grCube.load_file_path)[1])[0]
            grCube.plotItem = ui_util.HoverableCurveItem(grCube.data_x, grCube.data_y, name=title)
            self.rightPanel.graphView.addItem(grCube.plotItem)
            grCube.chkbox_module = self.leftPanel.graph_list_area.add_module(grCube.load_file_path)
            grCube.set_color(idx)
            grCube.chkbox_module.toggled.connect(grCube.visible_toggle)
            grCube.chkbox_module.toggled.connect(self.update_graph)
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
            if self.leftPanel.graph_ops_select_area.chkbox_std.isChecked():
                self.calculate_std()
            else:
                self.std_plot.hide()


    def x_axis_changed(self):
        current_x_axis = self.leftPanel.graph_x_data_select_area.cmb_x_data.currentText()
        btn_grp = self.leftPanel.graph_y_data_select_area.radio_grp
        btn_grp.blockSignals(True)
        buttons = btn_grp.buttons()
        [radio.blockSignals(True) for radio in buttons]
        for idx, radio in enumerate(buttons):
            if current_x_axis == radio.text():
                radio.setEnabled(False)
                if radio.isChecked():
                    buttons[np.mod(idx+1,len(buttons))].setChecked(True)
            else:
                radio.setEnabled(True)
        btn_grp.blockSignals(False)
        [radio.blockSignals(False) for radio in buttons]
        self.change_range()
        self.set_data()
        self.update_graph()

        if current_x_axis == "None":
            return
        shared = set(self.grCubes[0].pd_data[current_x_axis].to_list())
        unioned = set(self.grCubes[0].pd_data[current_x_axis].to_list())
        for grCube in self.grCubes:
            shared = set.intersection(shared, set(grCube.pd_data[current_x_axis].to_list()))
            unioned = set.union(unioned, set(grCube.pd_data[current_x_axis].to_list()))
        if (len(shared) / len(unioned)) < 0.7:
            QMessageBox.warning(self,"Warning","axis seems doesn't match \nShared percentage of axis is below 70%")

        self.rightPanel.graphView.autoRange()

    def y_axis_changed(self, state):
        if not state:
            return
        self.set_data()
        self.update_graph()
        self.rightPanel.graphView.autoRange()

    def update_graph(self):
        l = self.leftPanel.graph_range_area.spinbox_range_min.value()
        r = self.leftPanel.graph_range_area.spinbox_range_max.value()

        if len(self.grCubes) == 0:
            return
        if not hasattr(self.grCubes[0], 'data_x'):
            return
        if self.grCubes[0].plotItem is None:
            return

        for grCube in self.grCubes:
            idx_l, value_l = util.find_nearest(grCube.data_x, l)
            idx_r, value_r = util.find_nearest(grCube.data_x, r)
            range_slice = slice(idx_l, idx_r + 1)
            grCube.plotItem.setData(grCube.data_x[range_slice],grCube.data_y[range_slice])

        # average
        if self.leftPanel.graph_ops_select_area.chkbox_average.isChecked():
            self.average_plot.setVisible(True)
            self.calculate_average()
            self.average_plot.setData(self.avg_x[range_slice], self.avg_y[range_slice])
        else:
            self.average_plot.setVisible(False)
        # std
        if self.leftPanel.graph_ops_select_area.chkbox_std.isChecked():
            self.std_plot.setVisible(True)
            self.calculate_std()
            self.std_plot.setData(self.std_x[range_slice], self.std_y[range_slice])
        else:
            self.std_plot.setVisible(False)
        print("update graph")

    def data_cut(self, cubes):
        min_length = min([len(grCube.pd_data) for grCube in cubes])
        for grCube in cubes:
            grCube.pd_data = grCube.pd_data[0:min_length]

    def change_range(self):
        x_axis = self.leftPanel.graph_x_data_select_area.cmb_x_data.currentText()
        if x_axis == "None":
            l = 0
            r = len(self.grCubes[0].pd_data)-1
            for grCube in self.grCubes:
                r = max(r,len(grCube.pd_data)-1)
        else:
            nparr = self.grCubes[0].pd_data[x_axis].to_numpy()
            l = nparr.min()
            r = nparr.max()
            for grCube in self.grCubes:
                nparr = grCube.pd_data[x_axis].to_numpy()
                l = min(l,nparr.min())
                r = max(r,nparr.max())
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
        radio_btn = self.leftPanel.graph_y_data_select_area.radio_grp.checkedButton()
        if radio_btn is None:
            return
        for grcube in self.grCubes:
            str_xaxis = self.leftPanel.graph_x_data_select_area.cmb_x_data.currentText()
            if str_xaxis == "None":
                grcube.data_x = np.arange(0,len(grcube.pd_data))
            else:
                grcube.data_x = grcube.pd_data[str_xaxis].to_numpy()
            str_yaxis = radio_btn.text()
            grcube.data_y = grcube.pd_data[str_yaxis].to_numpy()

    def open_gr_clicked(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self,'open')
        self.setWindowTitle(path)
        if path == "":
            return

        data_r_files, data_azav_files = self.get_gr_files(path)
        if len(data_r_files) == 0:
            QMessageBox.about(self,"Error","No data detected on the folder")

        dcs = []
        for i in range(len(data_r_files)):
            dc = GrCube(data_r_files[i])
            dc.azavg_file_path = data_azav_files[i]
            dcs.append(dc)

        self.menu_save_intensity_profile_avg.setEnabled(True)
        self.open_stack(dcs)


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
        df_sum = pd.DataFrame()
        str_xaxis = self.leftPanel.graph_x_data_select_area.cmb_x_data.currentText()
        str_yaxis = self.leftPanel.graph_y_data_select_area.radio_grp.checkedButton().text()
        for grCube in self.grCubes:
            if grCube.chkbox_module.isChecked() == False:
                continue
            if str_xaxis == 'None':
                df = pd.DataFrame(grCube.pd_data[str_yaxis])
            else:
                df = pd.DataFrame(grCube.pd_data[str_yaxis].values,index=grCube.pd_data[str_xaxis])
            df_sum = pd.concat([df_sum,df],axis=1)
        df_mean = df_sum.mean(axis=1)
        self.avg_x = df_mean.index.to_numpy()
        self.avg_y = df_mean.to_numpy().squeeze()
        return self.avg_x, self.avg_y

    def calculate_std(self):
        df_sum = pd.DataFrame()
        str_xaxis = self.leftPanel.graph_x_data_select_area.cmb_x_data.currentText()
        str_yaxis = self.leftPanel.graph_y_data_select_area.radio_grp.checkedButton().text()
        for grCube in self.grCubes:
            if grCube.chkbox_module.isChecked() == False:
                continue
            if str_xaxis == 'None':
                df = pd.DataFrame(grCube.pd_data[str_yaxis])
            else:
                df = pd.DataFrame(grCube.pd_data[str_yaxis].values,index=grCube.pd_data[str_xaxis])
            df_sum = pd.concat([df_sum,df],axis=1)
        df_std = df_sum.std(axis=1)
        self.std_x = df_std.index.to_numpy()
        self.std_y = df_std.to_numpy().squeeze()
        return self.std_x, self.std_y

    def create_menu_bar(self):
        self.mainWindow = self.mainWindow
        self.menubar = self.mainWindow.menuBar()
        self.menubar.setNativeMenuBar(False)

        self.menu_open_csv = QtWidgets.QAction("csv", self.mainWindow)
        self.menu_open_txt = QtWidgets.QAction("txt", self.mainWindow)
        self.menu_open_preset = QtWidgets.QAction("preset", self.mainWindow)
        self.menu_open_preset.setDisabled(True)
        self.menu_open_gr = QtWidgets.QAction("gr", self.mainWindow)
        self.menu_open_custom = QtWidgets.QAction("Custom name", self.mainWindow)



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

        self.menu_save_current = QtWidgets.QAction("Save current graphs", self.mainWindow)
        self.menu_save_intensity_profile_avg = QtWidgets.QAction("Save intensity profile averaging", self.mainWindow)
        save_menu = self.menubar.addMenu("     &Save     ")
        save_menu.addAction(self.menu_save_current)
        save_menu.addAction(self.menu_save_intensity_profile_avg)
        self.menu_save_intensity_profile_avg.setEnabled(False)

        return self.menubar


class GrCube(datacube.DataCube):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_file_path = None
        self.color: QColor
        self.color = None
        self.plotItem = None
        self.chkbox_module:GraphCheckBox
        self.chkbox_module = None

    def clear(self):
        self.plotItem.clear()
        self.chkbox_module.deleteLater()
        self.chkbox_module = None

    def set_color(self, int):
        ## color ##
        self.color = pg.intColor(int, minValue=200, alpha=255)
        # self.color_dark = self.color.darker(50)
        # self.color_bright = self.color.lighter(50)
        self.color_txt = "rgba({}, {}, {}, {});".format(self.color.red(), self.color.green(), self.color.blue(), self.color.alpha())

        ## pen ##
        self.enter_pen = pg.mkPen(self.color, width=highlight_pen_thickness)
        self.default_pen = pg.mkPen(self.color,width=default_pen_thickness)

        ## default chkbox ##
        self.styleSheet_default = "background-color: #444444;" \
                                  "padding-top: 10px;"\
                                  "padding-bottom: 10px;"\
                                  "color:{}".format(self.color_txt)
        self.styleSheet_highlight =  "background-color: #777777;"\
                                     "padding-top: 10px;"\
                                     "padding-bottom: 10px;"\
                                     "color:{}".format(self.color_txt)
        self.styleSheet_unuse =      "background-color: #111111;"\
                                     "padding-top: 10px;"\
                                     "padding-bottom: 10px;"\
                                     "color:{}".format(self.color_txt)
        self.styleSheet_default_using = self.styleSheet_default
        self.styleSheet_highlight_using = self.styleSheet_highlight

        self.chkbox_module.setStyleSheet(self.styleSheet_default_using)

        ## default graphPen ##
        self.plotItem.setPen(self.default_pen)


    def binding_event(self):
        # widget enter event
        self.chkbox_module.sigEntered.connect(self.hover_in)
        self.chkbox_module.sigLeaved.connect(self.hover_out)
        self.plotItem.sigCurveHovered.connect(self.hover_in)
        self.plotItem.sigCurveNotHovered.connect(self.hover_out)
        self.plotItem.sigCurveClicked.connect(lambda: self.chkbox_module.setChecked(False))

    def hover_in(self):
        ### graph ###
        self.plotItem.setPen(self.enter_pen)

        ### chkbox ###
        self.chkbox_module.setStyleSheet(self.styleSheet_highlight_using)


    def hover_out(self):
        if self.styleSheet_default_using is None:
            return
        self.chkbox_module.setStyleSheet(self.styleSheet_default_using)
        self.plotItem.setPen(self.default_pen)
        self.chkbox_module.setStyleSheet(self.styleSheet_default_using)
        pass

    def visible_toggle(self):
        if self.chkbox_module.isChecked():
            self.plotItem.show()
            self.styleSheet_default_using = self.styleSheet_default
            self.styleSheet_highlight_using = self.styleSheet_highlight
            self.chkbox_module.setStyleSheet(self.styleSheet_highlight_using)
        else:
            self.plotItem.hide()
            self.styleSheet_default_using = self.styleSheet_unuse
            self.styleSheet_highlight_using = self.styleSheet_unuse
            self.chkbox_module.setStyleSheet(self.styleSheet_default_using)


class GraphPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(5,10,10,10)
        self.setLayout(self.layout)
        self.graphView = ui_util.CoordinatesPlotWidget(setYScaling=False, button1mode=True)
        self.axis1 = pg.InfiniteLine(angle=0)
        self.graphView.addItem(self.axis1)
        self.legend = self.graphView.addLegend()
        self.legend.setEnabled(False)
        self.legend.setVisible(False)
        self.layout.addWidget(self.graphView)


class LeftPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.splitter_left_vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        if platform.system() == 'Darwin':
            self.layout.setSpacing(0)
        self.layout.setContentsMargins(10,10,5,10)

        self.graph_list_area = self.GraphListArea()
        self.graph_ops_select_area = self.GraphOpsArea()
        self.graph_y_data_select_area = self.GraphYDataSelectArea()
        self.graph_range_area = self.GraphRangeArea()
        self.graph_x_data_select_area = self.GraphXDataSelectArea()

        # self.splitter_left_vertical.addWidget(self.graph_list_area)
        # self.splitter_left_vertical.addWidget(self.graph_ops_select_area)
        # self.splitter_left_vertical.addWidget(self.graph_x_data_select_area)
        # self.splitter_left_vertical.addWidget(self.graph_y_data_select_area)
        # self.splitter_left_vertical.addWidget(self.graph_range_area)

        self.splitter_left_vertical.setStretchFactor(0, 5)
        # self.splitter_left_vertical.setStretchFactor(1, 5)
        # self.splitter_left_vertical.setStretchFactor(2, 5)
        # self.splitter_left_vertical.addWidget(self.btn_group)

        # self.graph_y_data_select_area.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
        #                                                  QtWidgets.QSizePolicy.Policy.Maximum))
        self.layout.addWidget(self.graph_list_area)
        self.layout.addWidget(self.graph_ops_select_area)
        self.layout.addWidget(self.graph_x_data_select_area)
        self.layout.addWidget(self.graph_y_data_select_area)
        self.layout.addWidget(self.graph_range_area)
        # self.layout.addWidget(self.splitter_left_vertical)

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
            self.chkbox_legend = QtWidgets.QCheckBox("Show legend")
            self.layout.addWidget(self.spinbox_range_min)
            self.layout.addWidget(self.spinbox_range_max)
            self.layout.addWidget(self.chkbox_legend)

    class GraphOpsArea(QtWidgets.QGroupBox):
        def __init__(self):
            super().__init__("Operation")
            self.layout = QtWidgets.QVBoxLayout()
            self.setLayout(self.layout)
            self.chkbox_average = QtWidgets.QCheckBox("Average")
            self.chkbox_std = QtWidgets.QCheckBox("Std")
            self.layout.addWidget(self.chkbox_average)
            self.layout.addWidget(self.chkbox_std)

    class GraphYDataSelectArea(QtWidgets.QGroupBox):
        def __init__(self):
            super().__init__("Y data")
            self.layout = QtWidgets.QVBoxLayout()
            self.setLayout(self.layout)
            # self.layout.addWidget(QtWidgets.QCheckBox("Hdo"))
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

            self.setMinimumSize(200,300)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.setWidgetResizable(True)

        def add_module(self, text):
            print("module:",text)
            module = GraphCheckBox()
            module.setText(text)
            self.graph_group_widget.layout.insertWidget(0, module)
            return module


class GraphCheckBox(QtWidgets.QCheckBox):
    sigEntered = QtCore.pyqtSignal(object, object)
    sigLeaved = QtCore.pyqtSignal(object, object)
    def __init__(self):
        super().__init__()
        self.setChecked(True)

    def enterEvent(self, a0: QtCore.QEvent) -> None:
        super().enterEvent(a0)
        self.sigEntered.emit(self,a0)

    def leaveEvent(self, a0: QtCore.QEvent) -> None:
        super().leaveEvent(a0)
        self.sigLeaved.emit(self,a0)




if __name__ == "__main__":
    qtapp = QtWidgets.QApplication([])
    # QtWidgets.QMainWindow().show()
    viewer = DataViewer()
    viewer.show()
    qtapp.exec()