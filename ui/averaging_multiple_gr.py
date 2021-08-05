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

pg.setConfigOptions(antialias=True)

class Viewer(QtWidgets.QWidget):


    str_parameter = r"\Parameters.csv"
    str_data_r = r"\Data_r.csv"
    str_data_q = r"\Data_q.csv"

    def __init__(self):
        super().__init__()
        self.initui()
        self.grCubes = []
        self.avg_azavg = None
        self.binding()

    def initui(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.leftPanel = LeftPanel()
        self.rightPanel = GraphPanel()
        self.average_plot = self.rightPanel.graphView.plot(pen=pg.mkPen(255,255,255,width=5))
        self.layout.addWidget(self.leftPanel)
        self.layout.addWidget(self.rightPanel,1)

    def binding(self):
        self.leftPanel.btn_open_folder.clicked.connect(self.open_btn_clicked)
        self.leftPanel.btn_save_gr_avg.clicked.connect(self.save_gr_avg)
        self.leftPanel.btn_save_intensity_avg.clicked.connect(self.save_intensity_avg)
        self.leftPanel.btn_open_analyzer.clicked.connect(self.open_analyzer)

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
        azavg_list = []
        shortest_end = 1000000
        for grCube in self.grCubes:
            azavg = np.loadtxt(grCube.data_azav_path)
            if shortest_end > len(azavg):
                shortest_end = len(azavg)
            azavg_list.append(azavg)
        azavg_list = [azavg[:shortest_end] for azavg in azavg_list]
        np.array(azavg_list)
        avg_azavg = np.average(np.array(azavg_list), axis=0).transpose()
        self.avg_azavg = avg_azavg
        np.savetxt(fp+".txt", avg_azavg)

    def open_analyzer(self):
        if self.grCubes is None or len(self.grCubes) is 0:
            pass
        else:
            azavg_list = []
            shortest_end = 1000000
            for grCube in self.grCubes:
                azavg = np.loadtxt(grCube.data_azav_path)
                if shortest_end > len(azavg):
                    shortest_end = len(azavg)
                azavg_list.append(azavg)
            azavg_list = [azavg[:shortest_end] for azavg in azavg_list]
            np.array(azavg_list)
            avg_azavg = np.average(np.array(azavg_list), axis=0).transpose()
            self.avg_azavg = avg_azavg

        analyzer_window = main.DataViewer()
        analyzer_window.show()
        if self.avg_azavg is not None:
            analyzer_window.main_window.menu_open_azavg_only(self.avg_azavg)


    def open_btn_clicked(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self,'open')
        self.setWindowTitle(path)
        if path == "":
            return

        for grCube in self.grCubes: grCube.clear()
        self.grCubes.clear()

        data_r_files, data_azav_files = self.get_csv_files_from_folder(path)
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
            grCube.project_folder = path
            grCube.data_r_path = data_r_file
            grCube.data_azav_path = data_azav_files[idx]
            color = grCube.color
            plot_ = self.rightPanel.graphView.plot(grCube.r,grCube.Gr,name=graph_name,pen=pg.mkPen(color))
            grCube.plotItem = plot_
            color_rgb_text = "rgb({}, {}, {});".format(color.red(),color.green(),color.blue())
            grCube.chkbox_module = self.leftPanel.graphGroup.add_module(os.path.split(data_r_file)[1],color_rgb_text)
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


    def get_csv_files_from_folder(self, folder):
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


class GrCube:
    def __init__(self):
        self.parameter_path = None
        self.data_r_path = None
        self.data_q_path = None
        self.data_azav_path = None
        self.color = None

        self.project_folder = None
        self.plotItem = None
        self.chkbox_module:GraphModule
        self.chkbox_module = None
        self.Gr = None


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
        self.graphView = ui_util.CoordinatesPlotWidget(title='G(r)')
        self.axis1 = pg.InfiniteLine(angle=0)
        self.graphView.addItem(self.axis1)
        self.layout.addWidget(self.graphView)




class LeftPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()


        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.btn_open_folder = QtWidgets.QPushButton("Open Folder")
        self.btn_save_gr_avg = QtWidgets.QPushButton("save G(r) avg")
        self.btn_save_intensity_avg = QtWidgets.QPushButton("save selected intensity avg")
        self.btn_open_analyzer = QtWidgets.QPushButton("Open analyzer")

        self.btn_group = QtWidgets.QWidget()
        self.btn_group.layout = QtWidgets.QHBoxLayout()
        self.btn_group.layout.setContentsMargins(0,0,0,0)
        self.btn_group.setLayout(self.btn_group.layout)
        self.btn_group.layout.addWidget(self.btn_open_folder)
        self.btn_group.layout.addWidget(self.btn_save_gr_avg)
        self.btn_group.layout.addWidget(self.btn_save_intensity_avg)
        self.btn_group.layout.addWidget(self.btn_open_analyzer)

        self.graphGroup = GraphGroup()



        self.layout.addWidget(self.btn_group)

        scrollArea = QtWidgets.QScrollArea()
        scrollArea.setWidget(self.graphGroup)
        scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollArea.setWidgetResizable(True)
        scrollArea.setFixedHeight(200)
        self.layout.addWidget(scrollArea)

class GraphGroup(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

    def add_module(self, text, color):
        module = GraphModule()
        module.set_text(text)
        module.set_color(color)
        self.layout.addWidget(module)
        return module



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
    viewer = Viewer()
    viewer.show()
    qtapp.exec()