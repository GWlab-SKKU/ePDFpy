import ui.ui_util as ui_util
from PyQt5 import QtCore, QtWidgets, QtGui
import numpy as np
from calculate.autofit import Autofit
import pyqtgraph as pg
import pickle

table_head_lst = ['min_q', 'max_q', 'fix_q', 'N', 'noise1', 'color']

idx_Min_pix = 0
idx_Max_pix = 1
idx_qk = 2
idx_N = 3
idx_Q = 4
idx_phiq = 5
idx_phiq_d = 6
idx_r = 7
idx_g = 8

color = [pg.intColor(i, minValue=200, alpha=255) for i in range(100)]

class MainWindowAdvancedFit(QtWidgets.QMainWindow):
    def __init__(self, dc, close_event):
        QtWidgets.QMainWindow.__init__(self)
        viewer = AdvancedFitWindow(dc, close_event, self)
        self.setCentralWidget(viewer)
        self.setStyleSheet(ui_util.get_style_sheet())
        self.resize(1200, 700)

class AdvancedFitWindow(QtWidgets.QWidget):
    def __init__(self, dc, close_event, mainWindow):
        QtWidgets.QWidget.__init__(self)
        self.mainWindow = mainWindow
        self.dc = dc
        self.close_event = close_event
        splitter_vertical = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter_left_horizontal = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter_right_horizontal = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitter_vertical.addWidget(splitter_left_horizontal)
        splitter_vertical.addWidget(splitter_right_horizontal)

        self.gr_plot_lst = []
        self.phiq_plot_lst = []
        self.panel_control = self.ControlPanel()
        self.panel_table = self.TablePanel()
        self.panel_gr = self.GrPanel()
        self.panel_phiq = self.PhiPanel()

        splitter_left_horizontal.addWidget(self.panel_control)
        splitter_left_horizontal.addWidget(self.panel_table)
        splitter_right_horizontal.addWidget(self.panel_phiq)
        splitter_right_horizontal.addWidget(self.panel_gr)


        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(splitter_vertical)
        self.setLayout(layout)

        self.panel_control.btn_fit.clicked.connect(self.fit_clicked)
        self.panel_table.btn_select.clicked.connect(self.select)

        self.panel_table.table.itemSelectionChanged.connect(self.cell_clicked) # is duplicated for each run fit?

    def select(self):
        row = self.panel_table.table.currentRow()
        N = self.Candidates[row, idx_N]
        qk = self.Candidates[row, idx_qk]
        max_px = self.Candidates[row, idx_Max_pix]
        self.close_event(N,qk,max_px)
        self.mainWindow.close()

    def fit_clicked(self):
        Iq = self.dc.azavg
        qkran_start = self.panel_control.spinBox_range1_l.value()
        qkran_end = self.panel_control.spinBox_range1_r.value()
        # qkran_step = self.panel_control.spinBox_range1_r.value()
        qkran_step = self.panel_control.spinBox_range1_step.value()
        pixran_start = self.panel_control.spinBox_range2_l.value()
        pixran_end = self.panel_control.spinBox_range2_r.value()
        # pixran_end = self.panel_control.spinBox_range2_r.value()
        pixran_step = 5
        Elem = self.dc.element_nums
        Rat = self.dc.element_ratio
        pixel_start_n = self.dc.pixel_start_n
        Calibration_factor = self.dc.ds
        Damping_factor = self.dc.damping
        Noise_threshold = self.panel_control.spinBox_noise_thresholding.value()
        Select = 10
        TEM_voltage = self.dc.electron_voltage   #edited by MH 230119
        use_lobato = True if self.dc.scattering_factor =='Lobato' else False


        self.Candidates, qualitycheck, self.total_n = Autofit(Iq,qkran_start,qkran_end,qkran_step,pixran_start,pixran_end,pixran_step,Elem,Rat,
            pixel_start_n,Calibration_factor,Damping_factor,Noise_threshold,Select,use_lobato,TEM_voltage)

        # with open('candidates.p', 'rb') as file:
        #     self.Candidates = pickle.load(file)
        # qualitycheck, self.total_n = 10, 200

        self.panel_control.lbl_filtered_count.setText(f"{qualitycheck} / {self.total_n}", )

        self.draw_gr()
        self.draw_phiq()
        self.draw_table()

    def draw_gr(self):
        cnt = min(len(self.Candidates), self.panel_control.spinBox_result_count.value())
        [plot.clear() for plot in self.gr_plot_lst]
        self.gr_plot_lst.clear()
        for i in range(cnt):
            pen = pg.mkPen(color=color[i])
            plot = self.panel_gr.graph.plot()
            plot.setData(self.Candidates[i, idx_r], self.Candidates[i, idx_g], pen=pen)
            self.gr_plot_lst.append(plot)

    def draw_phiq(self):
        cnt = min(len(self.Candidates), self.panel_control.spinBox_result_count.value())
        [plot.clear() for plot in self.phiq_plot_lst]
        self.phiq_plot_lst.clear()
        for i in range(cnt):
            pen = pg.mkPen(color=color[i])
            plot = self.panel_phiq.graph.plot()
            plot.setData(self.Candidates[i, idx_Q], self.Candidates[i, idx_phiq], pen=pen)
            self.phiq_plot_lst.append(plot)


    def draw_table(self):
        self.panel_table.table.clear()
        self.panel_table.table.setHorizontalHeaderLabels(table_head_lst)
        column_idx_lst = [idx_Min_pix, idx_Max_pix, idx_qk, idx_N, 9]

        tbl = self.Candidates[0:min(len(self.Candidates), self.panel_control.spinBox_result_count.value()), column_idx_lst]
        idx_pair = []
        for row in np.arange(tbl.shape[0]):
            for column in np.arange(tbl.shape[1]):
                idx_pair.append([row, column])

        for row, column in idx_pair:
            self.panel_table.table.setItem(row, column, QtWidgets.QTableWidgetItem(str(tbl[row,column])))

        # color column
        for row in np.arange(tbl.shape[0]):
            self.panel_table.table.setItem(row, 5, QtWidgets.QTableWidgetItem(""))
            self.panel_table.table.item(row, 5).setBackground(QtGui.QColor(*color[row].getRgb()))

    def cell_clicked(self):
        row = self.panel_table.table.currentRow()
        if len(self.gr_plot_lst) == 0:
            return
        for irow in range(self.panel_control.spinBox_result_count.value()):
            color = pg.intColor(irow, minValue=200, alpha=255)
            if irow == row:
                pen = pg.mkPen(color=color, width=ui_util.highlight_pen_thickness)
                self.gr_plot_lst[irow].setPen(pen)
                self.phiq_plot_lst[irow].setPen(pen)
            else:
                pen = pg.mkPen(color=color, width=ui_util.default_pen_thickness)
                self.gr_plot_lst[irow].setPen(pen)
                self.phiq_plot_lst[irow].setPen(pen)


    class ControlPanel(QtWidgets.QGroupBox):
        def __init__(self):
            QtWidgets.QGroupBox.__init__(self, "Fitting Setting")
            layout = QtWidgets.QGridLayout()
            self.setLayout(layout)

            lbl_range1 = QtWidgets.QLabel("q_k range")
            lbl_range2 = QtWidgets.QLabel("pixel range")
            self.spinBox_range1_l = QtWidgets.QDoubleSpinBox()
            self.spinBox_range1_r = QtWidgets.QDoubleSpinBox()
            self.spinBox_range1_step = QtWidgets.QDoubleSpinBox()
            self.spinBox_range1_step.setValue(0.02)
            self.spinBox_range1_l.setValue(19)
            self.spinBox_range1_r.setValue(21)
            self.spinBox_range2_l = QtWidgets.QSpinBox()
            self.spinBox_range2_l.setMaximum(10e5)
            self.spinBox_range2_l.setValue(900)
            self.spinBox_range2_r = QtWidgets.QSpinBox()
            self.spinBox_range2_r.setMaximum(10e5)
            self.spinBox_range2_r.setValue(1150)
            self.spinBox_range2_step = QtWidgets.QSpinBox()
            self.spinBox_range2_step.setValue(5)
            lbl_noise_thresholding = QtWidgets.QLabel("Noise peak cut (%)")
            self.spinBox_noise_thresholding = QtWidgets.QDoubleSpinBox()
            self.spinBox_noise_thresholding.setValue(1)
            self.btn_fit = QtWidgets.QPushButton("Autofit")
            lbl_result_count = QtWidgets.QLabel("How many results?")
            self.lbl_filtered_text = QtWidgets.QLabel("Passed :")
            self.lbl_filtered_count = QtWidgets.QLabel("")
            self.lbl_filtered_text.setMaximumHeight(30)
            self.lbl_filtered_count.setMaximumHeight(30)
            self.spinBox_result_count = QtWidgets.QSpinBox()
            self.spinBox_result_count.setValue(5)
            self.spinBox_result_count.setMaximum(10)
            self.spinBox_result_count.setMinimum(1)

            layout.addWidget(lbl_range1, 0, 0)
            layout.addWidget(self.spinBox_range1_l, 0, 1)
            layout.addWidget(self.spinBox_range1_r, 0, 2)
            layout.addWidget(self.spinBox_range1_step, 0, 3)
            layout.addWidget(lbl_range2, 1, 0)
            layout.addWidget(self.spinBox_range2_l, 1, 1)
            layout.addWidget(self.spinBox_range2_r, 1, 2)
            layout.addWidget(self.spinBox_range2_step, 1, 3)

            layout.addWidget(lbl_noise_thresholding, 2, 0)
            layout.addWidget(self.spinBox_noise_thresholding, 2, 1)
            layout.addWidget(lbl_result_count, 3, 0)
            layout.addWidget(self.spinBox_result_count, 3, 1)
            layout.addWidget(self.btn_fit, 4, 0, 1, 3)
            layout.addWidget(self.lbl_filtered_text, 5, 0, 1, 1)
            layout.addWidget(self.lbl_filtered_count, 5, 1, 1, 1)

    class TablePanel(QtWidgets.QWidget):
        def __init__(self):
            QtWidgets.QWidget.__init__(self)
            self.table = QtWidgets.QTableWidget()
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(self.table)
            self.table.setRowCount(10)
            column_count = 6
            self.table.setColumnCount(column_count)
            for i in range(column_count):
                self.table.setColumnWidth(i,50)
            self.table.setHorizontalHeaderLabels(table_head_lst)
            self.table.setDragEnabled(False)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self.setLayout(layout)
            # self.table.setSelectionMode(QtGui.QAbstractItemView.SingleSelection)
            # self.table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            
            self.btn_select = QtWidgets.QPushButton("Select")
            layout.addWidget(self.btn_select)

    class GrPanel(QtWidgets.QWidget):
        def __init__(self):
            QtWidgets.QWidget.__init__(self)
            self.graph = ui_util.CoordinatesPlotWidget(title='G(r)', setYScaling=False, button1mode=True)
            layout = QtWidgets.QHBoxLayout()
            layout.addWidget(self.graph)
            self.setLayout(layout)
            axis = pg.InfiniteLine(angle=0)
            self.graph.addItem(axis)

    class PhiPanel(QtWidgets.QWidget):
        def __init__(self):
            QtWidgets.QWidget.__init__(self)
            self.graph = ui_util.CoordinatesPlotWidget(title='phi(q)', setYScaling=False, button1mode=True)
            layout = QtWidgets.QHBoxLayout()
            layout.addWidget(self.graph)
            self.setLayout(layout)
            axis = pg.InfiniteLine(angle=0)
            self.graph.addItem(axis)
