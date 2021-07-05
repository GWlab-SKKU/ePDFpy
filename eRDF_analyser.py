import typing

from PyQt5 import QtCore, QtWidgets, QtGui
class eRDF_analyser(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        print("init")
        self.setGeometry(300, 300, 300, 200)
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(QtWidgets.QPushButton("hello"))
        self.setLayout(self.layout)
        self.show()
