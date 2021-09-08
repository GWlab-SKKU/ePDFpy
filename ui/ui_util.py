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

        # self.cross_hair = self.plot()

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
        qp = self.plotItem.vb.mapSceneToView(ev.localPos())

        try:  # some qt version use different path
            modifiers = QtGui.QApplication.keyboardModifiers()
        except:
            modifiers = QtWidgets.QApplication.keyboardModifiers()

        if modifiers == QtCore.Qt.ShiftModifier:
            # shift click
            self.coor_update_toggle = not self.coor_update_toggle
        elif modifiers == QtCore.Qt.ControlModifier:
            # ctrl click
            if hasattr(self, 'crosshair_plot') and self.crosshair_plot is not None:
                self.removeItem(self.crosshair_plot)
                self.crosshair_plot = None
            else:
                self.crosshair_curve_dataItem, self.crosshair_idx = self.find_closest_coor(qp.x(), qp.y())
                self.create_cross_hair()
        elif modifiers == (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier):
            # shift + ctrl click
            pass
        else:
            pass
        # print(self.getPlotItem().dataItems[0].xDisp) # xData, yData, xDisp, yDisp
        return super().mousePressEvent(ev)

    def keyPressEvent(self, ev):
        if hasattr(self,'crosshair_plot') and self.crosshair_plot is not None:
            if ev.key() == QtCore.Qt.Key.Key_Right:
                self.move_cross_hair(+1)
            elif ev.key() == QtCore.Qt.Key.Key_Left:
                self.move_cross_hair(-1)
        super().keyPressEvent(ev)

    def create_cross_hair(self):
        self.crosshair_plot = self.plot(symbol='+', symbolSize=20, pen='b',symbolPen=pg.mkPen(width=1),symbolBrush=('b'),name="cross")
        legend = self.addLegend()
        self.crosshair_legend_label = legend.items[-1][-1]
        self.set_cross_hair_coordi()

    def move_cross_hair(self, x_increase):
        self.crosshair_idx = self.crosshair_idx+x_increase
        self.set_cross_hair_coordi()

    def set_cross_hair_coordi(self):
        x = self.crosshair_curve_dataItem.xData[self.crosshair_idx]
        y = self.crosshair_curve_dataItem.yData[self.crosshair_idx]
        self.crosshair_plot.setData([x],[y])
        str_x = str(np.round(x,2))
        str_y = str(np.round(y,2))
        self.crosshair_legend_label.setText("x:{},y:{}".format(str_x, str_y))

    def find_closest_coor(self, x,y):
        """
        :param x:
        :param y:
        :return: dataItem, idx
        """
        if len(self.getPlotItem().dataItems) == 0:
            return
        dataitem_idx_distance = []
        for dataItem in self.getPlotItem().dataItems:
            if dataItem.xData is None:
                continue
            x_data = np.asarray(dataItem.xData)
            y_data = np.asarray(dataItem.yData)
            distance_arr = np.abs(x_data-x)
            idx = distance_arr.argmin()
            distance = (x_data[idx]-x)**2 + (y_data[idx]-y)**2
            dataitem_idx_distance.append([dataItem,idx,distance])

        cloest_dataItem = dataitem_idx_distance[0]
        for dataItem, idx, distance in dataitem_idx_distance[1:]:
            if distance < cloest_dataItem[2]:
                cloest_dataItem = [dataItem, idx, distance]
        return cloest_dataItem[0],cloest_dataItem[1]

    def setYScaling(self, bool):
        if bool:
            self.sigXRangeChanged.connect(self.YScaling)
        else:
            self.sigXRangeChanged.disconnect()

    def YScaling(self):
        self.enableAutoRange(axis='y')
        self.setAutoVisible(y=True)

class IntensityPlotWidget(CoordinatesPlotWidget):
    def mousePressEvent(self, ev):
        if hasattr(self,'select_mode') and self.select_mode is True and ev.button() == QtCore.Qt.LeftButton:
            qp = self.plotItem.vb.mapSceneToView(ev.localPos())
            data = np.concatenate([self.first_dev_plot.getData()[0],self.second_dev_plot.getData()[0]])
            distance = np.abs(data - qp.x())
            idx = np.argmin(distance)
            print(data[idx])
            left, right = self.region.getRegion()
            self.region.setRegion([data[idx],right])
            self.select_event()

        # print(self.getPlotItem().dataItems[0].xDisp) # xData, yData, xDisp, yDisp
        return super().mousePressEvent(ev)

    def create_circle(self,first_dev,second_dev):
        if not hasattr(self,'first_dev_plot') or self.first_dev_plot is None:

            self.first_dev_plot = pg.ScatterPlotItem(size=10, brush='y')
            self.second_dev_plot = pg.ScatterPlotItem(size=10, brush='b')
            self.addItem(self.first_dev_plot)
            self.addItem(self.second_dev_plot)
            # self.first_dev_plot = self.plot(symbol='o', symbolSize=10, pen='b', symbolPen=pg.mkPen(width=1),
            #                                 symbolBrush=('y'), name="first derivative")
            # self.second_dev_plot = self.plot(symbol='o', symbolSize=10, pen='b', symbolPen=pg.mkPen(width=1),
            #                                 symbolBrush=('g'), name="second derivative")
        self.first_dev_plot.setData(first_dev[0], first_dev[1])
        self.second_dev_plot.setData(second_dev[0],second_dev[1])
        pass


def update_value(widget:QtWidgets.QWidget, value):
    """ update value without occuring signal """
    widget.blockSignals(True)
    if issubclass(type(widget),QtWidgets.QRadioButton):
        widget.setChecked(value)
    if issubclass(type(widget),QtWidgets.QDoubleSpinBox):
        widget.setValue(value)
    if issubclass(type(widget),QtWidgets.QSpinBox):
        widget.setValue(value)
    if issubclass(type(widget),pg.LinearRegionItem):
        widget.setRegion(value)
    widget.blockSignals(False)
