import typing

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QWidget
import numpy as np
from pyqtgraph.graphicsItems.LegendItem import LegendItem
import pyqtgraph as pg


class binding():
    def __init__(self, widget, event):
        self.functions = []
        self.event = event
        self.widget = widget
        pass
    def add_event(self, func):
        self.functions.append(func)
        pass

    def change_value(self,value):
        self.event.disconnect()
        self.widget.setValue(value)
        self.event.connect()


class DoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.validator = QRegExpValidator(QRegExp(r"[0-9]+[.][0-9]+"))
        # self.validator = QRegExpValidator(QRegExp(r"?[0-9]+\.?[0-9]+"))
        self.validator_int = QRegExpValidator(QRegExp(r"[0-9]+"))
        self.setDecimals(10)

    def validate(self, input: str, pos: int):
        if '.' in input:
            return self.validator.validate(input, pos)
        else:
            return self.validator_int.validate(input, pos)


    def textFromValue(self, v: float) -> str:
        # print("value:",v)
        # print("text:",super().textFromValue(v))
        str_v = str(v)
        return str(np.round(float(str_v),self.decimals()))

    def valueFromText(self, text: str) -> float:
        # print("text:",text)
        # print("value:",super().valueFromText(text))
        print("input:",str)
        print("output:", np.round(float(text),self.decimals()))
        return np.round(float(text),self.decimals())



class DoubleLineEdit(QtWidgets.QLineEdit):
    def __init__(self):
        super().__init__()
        self.validator = QRegExpValidator(QRegExp(r"[0-9]+[.]{0,1}[0-9]+"))

    def validate(self, input: str, pos: int):
        return self.validator.validate(input, pos)


class IntLineEdit(QtWidgets.QLineEdit):
    def __init__(self):
        super().__init__()
        self.validator = QRegExpValidator(QRegExp(r"[0-9]+"))

    def validate(self, input: str, pos: int):
        return self.validator.validate(input, pos)


class CoordinatesPlotWidget(pg.PlotWidget):
    def __init__(self, parent=None, background='default', offset=None, plotItem=None, **kargs):
        super().__init__(parent, background, plotItem, **kargs)

        # self.setRange(QRectF(-50, -50, 100, 100))
        # self.coor_label = pg.TextItem(text="x:{} \ny:{}".format(0, 0))
        # self.addItem(self.coor_label)
        # self.coor_label.setParentItem(self.getViewBox())
        # self.coor_label.setPos(10,10)


        # legend = self.addLegend()
        if offset is None:
            offset = (3,-3)

        legend = LegendItem(offset=offset)
        legend.setParentItem(self.getViewBox())

        style = pg.PlotDataItem()
        legend.addItem(style, 'A2')
        self.legend_labelitem = legend.getLabel(style)
        self.legend_labelitem.setText('x:0 y:0')
        self.coor_update_toggle = True

    def mouseMoveEvent(self, ev):
        if self.coor_update_toggle:
            qp = self.plotItem.vb.mapSceneToView(ev.localPos())
            x = str(qp.x())
            y = str(qp.y())
            x = x[:x.find('.') + 1] + x[x.find('.') + 1:][:4]
            y = y[:y.find('.') + 1] + y[y.find('.') + 1:][:4]
            self.legend_labelitem.setText("x:{} \ny:{}".format(x,y))
        # self.coor_label.setText("x:{} \ny:{}".format(str(qp.x())[:8],str(qp.y())[:8]))
        return super(CoordinatesPlotWidget, self).mouseMoveEvent(ev)

    def mousePressEvent(self, ev):
        # print(self.getPlotItem().dataItems[0].xDisp) # xData, yData, xDisp, yDisp
        self.coor_update_toggle = not self.coor_update_toggle
        return super().mousePressEvent(ev)

def update_value(widget:QtWidgets.QWidget, value):
    """ update value without occuring signal """
    widget.blockSignals(True)
    if issubclass(type(widget),QtWidgets.QRadioButton):
        widget.setChecked(value)
    if issubclass(type(widget),QtWidgets.QDoubleSpinBox):
        widget.setValue(value)
    if issubclass(type(widget),QtWidgets.QSpinBox):
        widget.setValue(value)

    widget.blockSignals(False)
