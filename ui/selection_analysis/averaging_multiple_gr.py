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

        self.splitter_horizontal = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter_horizontal.addWidget(self.leftPanel)
        self.splitter_horizontal.addWidget(self.rightPanel)


        self.menu_bar = self.create_menu_bar()

        self.layout.setContentsMargins(0,0,0,0)
        self.layout.setSpacing(0)
        self.layout.addWidget(self.menu_bar)
        self.layout.addWidget(self.splitter_horizontal)

    def binding(self):
        self.open_stack_gr.triggered.connect(self.open_btn_clicked)
        self.open_stack_csv.triggered.connect(self.open_stack_csv_clicked)
        # self.leftPanel.btn_group.btn_save_gr_avg.clicked.connect(self.save_gr_avg)
        # self.leftPanel.btn_group.btn_save_intensity_avg.clicked.connect(self.save_intensity_avg)
        # self.leftPanel.btn_group.btn_open_analyzer.clicked.connect(self.open_analyzer)

    def save_gr_avg(self):
        # calculate avg
        fp,ext = QtWidgets.QFileDialog.getSaveFileName(self, filter="CSV Files (*.csv);;All Files (*)")
        if fp == "":
            return
        df = pd.DataFrame({'r':self.grCubes[0].r,'Gr':self.gr_avg})
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

    def open_stack_csv_clicked(self):
        #### get files ####
        dcs = self.get_dcs_from_folder("*.csv")
        if dcs is None:
            return
        self.open_stack(dcs)

    def open_stack(self, dcs):
        if len(dcs) == 0:
            QMessageBox.about(self,"Error","No data detected on the folder")
        #### verify files ####
        if not self.verify_files(dcs):
            return

        # cut data
        self.data_cut(dcs)

        # x data
        columns = dcs[0].pd_data.columns.to_list()
        combo_lst = ["None"]
        combo_lst.extend(columns)
        self.leftPanel.graph_x_data_select_area.cmb_x_data.clear()
        self.leftPanel.graph_x_data_select_area.cmb_x_data.addItems(combo_lst)

        # y data
        self.leftPanel.graph_y_data_select_area.clear_radio()
        for clmn in columns:
            radio = self.leftPanel.graph_y_data_select_area.add_radio(clmn)
            radio.toggled.connect(self.clicked_y_radio)

        self.change_range()
        # columns_to_add = columns
        # if x_axis in columns_to_add:
        #     columns_to_add.remove(x_axis)
        # self.leftPanel.graph_column_select_area.add_radio()
        #### draw graphs ####
        # for dc in dcs:
        #     self.draw_graphs(dc)

    def clicked_y_radio(self):
        radio = self.sender()
        if radio.isChecked():
            pass

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
        self.leftPanel.graph_range_area.spinbox_range_min.setMinimum(l)
        self.leftPanel.graph_range_area.spinbox_range_max.setMinimum(l)
        self.leftPanel.graph_range_area.spinbox_range_min.setMaximum(r)
        self.leftPanel.graph_range_area.spinbox_range_max.setMaximum(r)

        ui_util.update_value(self.leftPanel.graph_range_area.spinbox_range_min, l)
        ui_util.update_value(self.leftPanel.graph_range_area.spinbox_range_max, r)



    def set_data(self):
        for grcube in self.grCubes:
            str_xaxis = self.leftPanel.graph_x_data_select_area.cmb_x_data.currentText()
            if str_xaxis == "None":
                grcube.data_x = np.arange(0,len(grcube.pd_data))
            else:
                grcube.data_x = grcube.pd_data[str_xaxis].to_numpy()
            str_yaxis = self.leftPanel.graph_y_data_select_area.radio_grp.checkedButton().text()
            grcube.data_y = grcube.pd_data[str_yaxis].to_numpy()

        ### range ###


    def reload_graph(self):
        for grcube in self.grCubes:
            grcube.plotItem.setData(grcube.data_x, grcube.data_y)
        self.rightPanel.graphView.autoRange()



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
            color_rgb_text = "rgb({}, {}, {});".format(color.red(),color.green(),color.blue())
            grCube.chkbox_module = self.leftPanel.graph_list_area.add_module(os.path.split(data_r_file)[1],color_rgb_text)
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
        dcs = [datacube.DataCube(file_path=str(file.absolute())) for file in files]
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
        Gr_list = []
        for grCube in self.grCubes:
            if grCube.plotItem.isVisible():
                Gr_list.append(grCube.Gr)
                r = grCube.r

        self.gr_avg = np.average(np.array(Gr_list).transpose(), axis=1)
        self.average_plot.setData(r, self.gr_avg)
        pass

    # def hovering_event(self):
    #     for grCube in self.grCubes:
    #         grCube.chkbox_module:GraphModule
    #         grCube.chkbox_module.enterEvent_list.append(grCube.plotItem)

    def create_menu_bar(self):
        self.mainWindow = self.mainWindow
        self.menubar = self.mainWindow.menuBar()

        self.open_file = QtWidgets.QAction("File", self.mainWindow)
        self.open_stack_csv = QtWidgets.QAction("csv", self.mainWindow)
        self.open_stack_txt = QtWidgets.QAction("txt", self.mainWindow)
        self.open_stack_preset = QtWidgets.QAction("preset", self.mainWindow)
        self.open_stack_gr = QtWidgets.QAction("gr", self.mainWindow)
        self.open_stack_custom_name = QtWidgets.QAction("Custom Name", self.mainWindow)

        open_menu = self.menubar.addMenu("     &Open     ")
        open_menu.addAction(self.open_file)
        open_stack = open_menu.addMenu("Stack")
        open_stack.addAction(self.open_stack_csv)
        open_stack.addAction(self.open_stack_txt)
        open_stack.addAction(self.open_stack_preset)
        open_stack.addAction(self.open_stack_gr)
        open_stack.addAction(self.open_stack_custom_name)
        open_menu.addSeparator()

        save_menu = self.menubar.addMenu("     &Save     ")
        return self.menubar


class GrCube(datacube.DataCube):
    def __init__(self):
        self.r_file_path = None
        self.color: QColor
        self.color = None
        self.plotItem = None
        self.chkbox_module:GraphModule
        self.chkbox_module = None

    def plot_show_hide(self):
        if self.chkbox_module.chkbox.isChecked():
            print(self.plotItem.isVisible())
            self.plotItem.show()
        else:
            print(self.plotItem.isVisible())
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
        self.chkbox_module.enterEvent_list.append(lambda :self.plotItem.setPen(enter_pen))
        self.chkbox_module.leaveEvent_list.append(lambda :self.plotItem.setPen(default_pen))



class GraphPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.graphView = ui_util.CoordinatesPlotWidget(title='G(r)', setYScaling=False)
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
        # self.splitter_left_vertical.addWidget(self.btn_group)

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


    class GraphListArea(QtWidgets.QScrollArea):
        def __init__(self):
            super().__init__()
            self.graph_group_widget = QtWidgets.QWidget()
            self.graph_group_widget.layout = QtWidgets.QVBoxLayout()
            self.graph_group_widget.setLayout(self.graph_group_widget.layout)
            self.setWidget(self.graph_group_widget)

            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
            self.setWidgetResizable(True)
            self.setFixedHeight(200)

        def add_module(self, text, color):
            print("module:",text)
            module = GraphModule()
            module.set_text(text)
            module.set_color(color)
            self.graph_group_widget.layout.addWidget(module)
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
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.setLayout(layout)
        self.chkbox = QtWidgets.QCheckBox("")
        self.chkbox.setChecked(True)
        self.textbox = QtWidgets.QPlainTextEdit("")
        self.textbox.setReadOnly(True)
        self.textbox.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.textbox.setMaximumHeight(20)
        layout.addWidget(self.chkbox)

        # layout.addWidget(self.textbox)

    # def set_text(self,text):
    #     self.textbox.setPlainText(text)

    def set_text(self,text):
        self.chkbox.setText(text)

    def set_color(self,color):
        self.setStyleSheet("background-color: {}".format(color))
        # self.chkbox.styleSheet()
        pass

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