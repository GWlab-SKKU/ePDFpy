import typing
from datacube import DataCube
import pyqtgraph as pg

from PyQt5 import QtCore, QtWidgets, QtGui
class rdf_analyse(QtWidgets.QWidget):
    def __init__(self, datacube: DataCube):
        QtWidgets.QWidget.__init__(self)
        print("init")
        self.datacube = datacube
        self.initui()


    def initui(self):
        self.setMinimumSize(500, 500)
        self.layout = QtWidgets.QGridLayout()
        self.layout.addWidget(QtWidgets.QPushButton("hello"))
        self.controlPanel = controlPanel()
        self.graph_Iq = pg.plot()
        self.graph_phiq = pg.plot()
        self.graph_
        self.layout.addWidget(self.controlPanel,0,0)
        self.layout.addWidget(self.controlPanel,1,0)
        self.setLayout(self.layout)
        self.show()

    def update_graph(self):
        pass

class controlPanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)


