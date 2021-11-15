import typing

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QWidget
import numpy as np
from pyqtgraph.graphicsItems.LegendItem import LegendItem
import pyqtgraph as pg
from calculate.pdf_calculator import pixel_to_q, q_to_pixel
import platform
import definitions


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
    def __init__(self, setYScaling=False, parent=None, background='default', offset=None, plotItem=None, button1mode=False, **kargs):
        super().__init__(parent, background, plotItem, **kargs)

        # self.setRange(QRectF(-50, -50, 100, 100))
        # self.coor_label = pg.TextItem(text="x:{} \ny:{}".format(0, 0))
        # self.addItem(self.coor_label)
        # self.coor_label.setParentItem(self.getViewBox())
        # self.coor_label.setPos(10,10)
        # self.cross_hair = self.plot()
        self.crosshair_plot = None
        self.graph_legend = self.getPlotItem().addLegend(offset=(-3, 3))

        self.setBackground('#2a2a2a')

        # legend = self.addLegend()
        if offset is None:
            offset = (3,-3)

        legend = LegendItem(offset=offset)
        legend.setParentItem(self.getViewBox())

        style = pg.PlotDataItem(pen=(0,0,0,0))  # make transparent
        legend.addItem(style, 'coordi')
        self.legend_labelitem = legend.getLabel(style)
        self.legend_labelitem.setText('x:0 y:0')
        self.coor_update_toggle = True

        if setYScaling:
            self.sigXRangeChanged.connect(self.YScaling)

        if button1mode:
            self.plotItem.vb.setLeftButtonAction('rect')

    def mouseMoveEvent(self, ev):
        if self.coor_update_toggle:
            qp = self.plotItem.vb.mapSceneToView(ev.localPos())
            x = str(qp.x())
            y = str(qp.y())
            x = x[:x.find('.') + 1] + x[x.find('.') + 1:][:4]
            y = y[:y.find('.') + 1] + y[y.find('.') + 1:][:4]
            self.legend_labelitem.setText("x:{} \ny:{}".format(x,y))
        # self.coor_label.setText("x:{} \ny:{}".format(str(qp.x())[:8],str(qp.y())[:8]))
        return super().mouseMoveEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        mouseMode = self.getPlotItem().getViewBox().getState()['mouseMode']
        if mouseMode == 1:
            # 1 button mode
            self.getPlotItem().getViewBox().autoRange()
        elif mouseMode == 3:
            # 3 button mode
            try:  # some qt version use different path
                modifiers = QtGui.QApplication.keyboardModifiers()
            except:
                modifiers = QtWidgets.QApplication.keyboardModifiers()

            qp = self.getPlotItem().getViewBox().mapSceneToView(ev.localPos())

            if modifiers == QtCore.Qt.AltModifier:
                # Zoom Out
                self.getPlotItem().getViewBox().scaleBy(x=3, y=3)
            else:
                # Zoom In
                self.getPlotItem().getViewBox().scaleBy(x=1/3, y=1/3)

            vr = self.getPlotItem().getViewBox().targetRect()
            center = self.getPlotItem().getViewBox().rect().center()
            center = self.getPlotItem().getViewBox().mapSceneToView(center)

            diff_x, diff_y = center.x() - qp.x(), center.y() - qp.y()
            x = vr.left() - diff_x, vr.right() - diff_x
            y = vr.top() - diff_y, vr.bottom() - diff_y
            self.setRange(xRange=x, yRange=y, padding=0)

        super().mouseDoubleClickEvent(ev)

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
                self.crosshair_legend.hide()
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
            if not (ev.modifiers() & (QtCore.Qt.Modifier.CTRL | QtCore.Qt.Modifier.SHIFT | QtCore.Qt.Modifier.ALT)):
                if ev.key() == QtCore.Qt.Key.Key_Right:
                    self.move_cross_hair(+1)
                elif ev.key() == QtCore.Qt.Key.Key_Left:
                    self.move_cross_hair(-1)
        super().keyPressEvent(ev)

    def create_cross_hair(self):
        if platform.system() == 'Darwin':
            args = {'symbol':'+', 'symbolSize':20, 'pen':'b','symbolPen':pg.mkPen(width=1),'symbolBrush':('b'),'name':"cross"}
            label_text_size = '15pt'
        else:
            args = {'symbol':'+', 'symbolSize':20, 'pen': 'b', 'symbolPen': pg.mkPen(width=1), 'symbolBrush': ('b'),'name':"cross"}
            label_text_size = '9pt'

        if self.crosshair_plot is None:
            self.crosshair_plot = self.plot(**args)
            self.graph_legend.removeItem(self.crosshair_plot)

        offset = (-3,-3)
        self.crosshair_legend = LegendItem(offset=offset, labelTextSize=label_text_size)
        self.crosshair_legend.setParentItem(self.getViewBox())

        style = pg.PlotDataItem(**args)
        self.crosshair_legend.addItem(style, 'cross')

        self.crosshair_legend_label = self.crosshair_legend.getLabel(style)
        self.crosshair_legend_label.setText('x:0 y:0')

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

    def save_intensity_avg(self):
        fp, ext = QtWidgets.QFileDialog.getSaveFileName(self, filter="text file (*.txt);;All Files (*)")
        if fp == "":
            return
        azavg_list = []
        shortest_end = 1000000
        for grCube in self.grCubes:
            azavg = np.loadtxt(grCube.azavg_file_path)
            if shortest_end > len(azavg):
                shortest_end = len(azavg)
            azavg_list.append(azavg)
        azavg_list = [azavg[:shortest_end] for azavg in azavg_list]
        avg_azavg = np.average(np.array(azavg_list), axis=0).transpose()
        np.savetxt(fp+".txt", avg_azavg)


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
        # else:
        #     self.sigXRangeChanged.disconnect()

    def YScaling(self):
        self.enableAutoRange(axis='y')
        self.setAutoVisible(y=True)

class HoverableCurveItem(pg.PlotCurveItem):
    # sigCurveHovered = QtCore.Signal(object, object)
    # sigCurveNotHovered = QtCore.Signal(object, object)
    # sigCurveClicked = QtCore.Signal(object, object)
    sigCurveHovered = QtCore.pyqtSignal(object, object)
    sigCurveNotHovered = QtCore.pyqtSignal(object, object)
    sigCurveClicked = QtCore.pyqtSignal(object, object)

    def __init__(self, hoverable=True, *args, **kwargs):
        super(HoverableCurveItem, self).__init__(*args, **kwargs)

    def hoverEvent(self, ev):
        if self.mouseShape().contains(ev.pos()):
            self.sigCurveHovered.emit(self, ev)
        else:
            self.sigCurveNotHovered.emit(self, ev)

    def mouseClickEvent(self, ev):
        if self.mouseShape().contains(ev.pos()):
            self.sigCurveClicked.emit(self, ev)
        return super().mouseClickEvent(ev)



class IntensityPlotWidget(CoordinatesPlotWidget):
    def mousePressEvent(self, ev):
        if hasattr(self,'select_mode') and self.select_mode is True and ev.button() == QtCore.Qt.LeftButton:
            qp = self.plotItem.vb.mapSceneToView(ev.localPos())
            data = np.concatenate([self.first_dev_plot.getData()[0],self.second_dev_plot.getData()[0]])
            distance = np.abs(data - qp.x())
            idx = np.argmin(distance)
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
    if issubclass(type(widget),QtWidgets.QComboBox):
        widget.setCurrentIndex(value)
    widget.blockSignals(False)




class QHLine(QtWidgets.QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.HLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)

class QVLine(QtWidgets.QFrame):
    def __init__(self):
        super(QVLine, self).__init__()
        self.setFrameShape(QtWidgets.QFrame.VLine)
        self.setFrameShadow(QtWidgets.QFrame.Sunken)


class ProfileGraphPanel(QtWidgets.QWidget):
    def __init__(self, title):
        QtWidgets.QWidget.__init__(self)
        self.imageView = pg.ImageView()
        self.datacube = None
        # self.plot_azav = pg.PlotWidget(title='azimuthal average')
        self.plotWidget = IntensityPlotWidget(title=title)
        self.plotWidget.setYScaling(True)
        self.setting = self.Setting()

        self.region = pg.LinearRegionItem([0, 100])

        self.plotWidget.region = self.region
        self.plotWidget.addItem(self.region)

        # self.legend = self.plotWidget.addLegend(offset=(-30, 30))
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.setting)
        self.layout.addWidget(self.plotWidget)
        self.layout.setContentsMargins(5,5,5,5)
        self.setLayout(self.layout)

        # self.setMaximumHeight(300)
        self.sig_binding()

        self.integer = False

    def hide_region(self):
        if self.setting.hide_checkBox.isChecked():
            self.region.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 50)))
            self.region.setHoverBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 100)))
            self.region.setMovable(True)
        else:
            self.region.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 0)))
            self.region.setHoverBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 0)))
            self.region.setMovable(False)
        self.update()

    def sig_binding(self):
        self.setting.hide_checkBox.clicked.connect(self.hide_region)
        self.setting.button_start.clicked.connect(self.btn_range_start_clicked)
        self.setting.button_all.clicked.connect(self.btn_range_all_clicked)
        self.setting.button_end.clicked.connect(self.btn_range_end_clicked)
        self.setting.spinBox_range_left.valueChanged.connect(self.dialog_to_range)
        self.setting.spinBox_range_right.valueChanged.connect(self.dialog_to_range)
        self.region.sigRegionChangeFinished.connect(self.range_to_dialog)

    def dialog_to_range(self):
        left, right = (self.setting.spinBox_range_left.value(),self.setting.spinBox_range_right.value())
        if self.integer:
            left = int(np.round(left))
            right = int(np.round(right))
        update_value(self.region, [left, right])
        self.datacube.pixel_start_n = q_to_pixel(left,self.datacube.ds)
        self.datacube.pixel_end_n = q_to_pixel(right,self.datacube.ds)
        self.setting.lbl_pixel_range.setText("({},{})".format(self.datacube.pixel_start_n,self.datacube.pixel_end_n))

    def range_to_dialog(self):
        left, right = self.region.getRegion()
        if self.integer:
            left = int(np.round(left))
            right = int(np.round(right))
        maxes = [dataItem.xData[-1] for dataItem in self.plotWidget.getPlotItem().dataItems]
        max = np.max(maxes)
        if right > max:
            right = max
        if left < 0:
            left = 0
        update_value(self.region, [left, right])
        update_value(self.setting.spinBox_range_left, left)
        update_value(self.setting.spinBox_range_right, right)
        self.datacube.pixel_start_n = q_to_pixel(left,self.datacube.ds)
        self.datacube.pixel_end_n = q_to_pixel(right,self.datacube.ds)
        self.setting.lbl_pixel_range.setText("({},{})".format(self.datacube.pixel_start_n, self.datacube.pixel_end_n))

    def btn_range_start_clicked(self):
        # left = q_range_selector.find_first_nonzero_idx(self.dc.azavg)
        left = self.setting.spinBox_range_left.value()
        right = self.setting.spinBox_range_right.value()

        l = left
        r = left + int((right - left) / 4)
        # r = left + int((len(self.dc.azavg) - left) / 4)
        # print("left {}, right {}".format(l, r))
        # mx = np.max(self.dc.azavg[l:r])
        # mn = np.min(self.dc.azavg[l:r])
        self.plotWidget.setXRange(l, r, padding=0.1)
        # self.graphPanel.plot_azav.setYRange(mn, mx, padding=0.1)
        # print(self.graphPanel.plot_azav.viewRange())

    def btn_range_all_clicked(self):
        self.plotWidget.autoRange()

    def btn_range_end_clicked(self):
        left = self.setting.spinBox_range_left.value()
        right = self.setting.spinBox_range_right.value()
        l = right-int((right - left) / 4)
        r = right
        # mx = np.max(self.dc.azavg[l:r])
        # mn = np.min(self.dc.azavg[l:r])
        self.plotWidget.setXRange(l, r, padding=0.1)
        # self.graphPanel.plot_azav.setYRange(mn, mx, padding=0.1)

    def update_graph(self, dat):
        # self.plotWindow.layout.setSpacing(0)
        # self.plotWindow.layout.setContentsMargins(0,0,0,0)
        # self.plot_azav = pg.PlotWidget(title='azimuthal average')
        # self.plotWindow.layout.addWidget(self.plot_azav)
        # self.plotWindow.setLayout(self.plotWindow.layout)
        self.plot_azav_curr.setData(dat)
        # self.plotWindow.resize(1000,350)

    class Setting(QtWidgets.QGroupBox):
        def __init__(self):
            QtWidgets.QGroupBox.__init__(self)
            self.layout = QtWidgets.QHBoxLayout()
            self.setLayout(self.layout)
            # remove title region
            self.setStyleSheet("QGroupBox{padding-top:0px; margin-top:0px}")

            # self.button_grp_widget.layout.addStretch(1)
            self.hide_checkBox = QtWidgets.QCheckBox()
            self.hide_checkBox.setObjectName("CheckVisible")
            # self.hide_checkBox.setFixedSize(20,20)
            self.hide_checkBox.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,QtWidgets.QSizePolicy.Policy.Fixed)
            self.hide_checkBox.setChecked(True)
            self.lbl_range = QtWidgets.QLabel("Q range")
            self.lbl_pixel_range = QtWidgets.QLabel("")
            self.spinBox_range_left = QtWidgets.QDoubleSpinBox()
            self.spinBox_range_left.setDecimals(3)
            self.spinBox_range_right = QtWidgets.QDoubleSpinBox()
            self.spinBox_range_right.setDecimals(3)
            self.layout.addWidget(self.hide_checkBox)
            self.layout.addWidget(self.lbl_range)
            self.layout.addWidget(self.lbl_pixel_range)
            self.layout.addWidget(self.spinBox_range_left)
            self.layout.addWidget(self.spinBox_range_right)
            self.layout.addWidget(QVLine())

            maximum_width = 40
            self.button_start = QtWidgets.QPushButton("╟─")
            self.button_start.setMaximumWidth(maximum_width)
            self.button_start.setSizePolicy(QtWidgets.QSizePolicy.Fixed,QtWidgets.QSizePolicy.Fixed)
            self.button_all = QtWidgets.QPushButton("├─┤")
            self.button_all.setMaximumWidth(maximum_width)
            self.button_end = QtWidgets.QPushButton("─╢")
            self.button_end.setMaximumWidth(maximum_width)
            self.button_select = QtWidgets.QPushButton("Select")
            self.button_select.setDisabled(True)
            self.layout.addWidget(self.button_start)
            self.layout.addWidget(self.button_all)
            self.layout.addWidget(self.button_end)
            self.layout.addWidget(self.button_select)
            # self.button_grp_widget.layout.addStretch(1)

def get_style_sheet(template=None):
    style_sheet = open(definitions.THEME_PATH, 'r').read()+open(definitions.STYLE_PATH, 'r').read()
    style_sheet = style_sheet.replace("image: url(","image: url("+definitions.ROOT_DIR+"/")
    return style_sheet

def get_style_sheet_dark():
    style_sheet = open(definitions.THEME_PATH2, 'r').read()+open(definitions.STYLE_PATH, 'r').read()
    style_sheet = style_sheet.replace("image: url(","image: url("+definitions.ROOT_DIR+"/")
    return style_sheet

