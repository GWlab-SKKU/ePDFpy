from PyQt5 import QtWidgets
import ui.main as main
import pyqtgraph as pg
import os
import glob
import pandas as pd
from pathlib import Path
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

class Viewer(QtWidgets.QWidget):

    str_parameter = r"\Parameters.csv"
    str_data_r = r"\Data_r.csv"
    str_data_q = r"\Data_q.csv"

    def __init__(self):
        super().__init__()
        self.initui()
        self.grCubes = []
        self.binding()

    def initui(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.leftPanel = LeftPanel()
        self.rightPanel = GraphPanel()
        self.layout.addWidget(self.leftPanel)
        self.layout.addWidget(self.rightPanel)

    def binding(self):
        self.leftPanel.btn_open_folder.clicked.connect(self.open_btn_clicked)

    def open_btn_clicked(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self,'open')
        self.setWindowTitle(path)
        if path == "":
            return
        data_r_files = self.get_csv_files_from_folder(path)
        if len(data_r_files) == 0 :
            QMessageBox.about(self,"Error","No data detected on the folder")
        ################## MAIN ####################
        for idx, data_r_file in enumerate(data_r_files):
            grCube = GrCube()
            grCube.unit_path = data_r_file

            # graph name
            graph_name = data_r_file[data_r_file.rfind("_")+1:data_r_file.find("Data_r.csv")]

            # load and plot
            r, Gr = self.load_Gr_file(data_r_file)
            plot_ = self.rightPanel.graphView.plot(r,Gr,name=graph_name,pen=pg.mkPen(pg.intColor(idx)))
            grCube.plotItem = plot_
            grCube.chkbox_module = self.leftPanel.graphGroup.add_module(os.path.split(data_r_file)[1],"yellow")
            print(grCube.chkbox_module.chkbox)
            grCube.chkbox_module.chkbox.clicked.connect(grCube.plot_show_hide)
            self.grCubes.append(grCube)




    def load_Gr_file(self, path):
        df = pd.read_csv(path)
        r = df['r'].to_numpy()
        Gr = df['Gr'].to_numpy()
        return r, Gr


    def get_csv_files_from_folder(self, folder):
        csvfiles = Path(folder).rglob("*Data_r.csv")
        fps = []
        for file in csvfiles:
            if file.name in ["diagonal.csv", "diagonal_1.csv", "line.csv"]:
                continue
            if "r30" in file.name:
                continue
            fps.append(str(file.absolute()))
        return fps


class GrCube:
    def __init__(self):
        self.parameter_path = None
        self.data_r_path = None
        self.data_q_path = None

        self.selected_folder = None
        self.unit_path = None
        self.unit_name = None
        self.Gr_path = None
        self.plotItem = None
        self.chkbox_module = None
        self.Gr = None

    def plot_show_hide(self):
        if self.chkbox_module.chkbox.isChecked():
            print("show, ", self)
            self.plotItem.show()
        else:
            print("hide",self)
            self.plotItem.hide()


class GraphPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.graphView = pg.PlotWidget(title='G(r)')
        self.layout.addWidget(self.graphView)

class LeftPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.btn_open_folder = QtWidgets.QPushButton("Open Folder")
        self.graphGroup = GraphGroup()

        self.layout.addWidget(self.btn_open_folder)
        self.layout.addWidget(self.graphGroup)

class GraphGroup(QtWidgets.QGroupBox):
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
        layout.addWidget(self.textbox)

    def set_text(self,text):
        self.textbox.setPlainText(text)
        pass

    def set_color(self,color):
        self.chkbox.setStyleSheet("background-color: {}".format("yellow"))
        # self.chkbox.styleSheet()
        pass



if __name__ == "__main__":
    qtapp = QtWidgets.QApplication([])
    # QtWidgets.QMainWindow().show()
    viewer = Viewer()
    viewer.show()
    qtapp.exec()