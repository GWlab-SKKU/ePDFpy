import typing

import file
from datacube import DataCube
import pyqtgraph as pg
import util
from calculate import rdf_calculator
from PyQt5.QtWidgets import QMessageBox
import ui.ui_util as ui_util
from PyQt5 import QtCore, QtWidgets, QtGui



class rdf_analyse(QtWidgets.QMainWindow):
    def __init__(self, datacube):
        super().__init__()
        self.datacube = datacube
        self.datacube.analyser = self
        if self.datacube.element_nums is None:
            self.datacube.element_nums = []
            self.datacube.element_ratio = []
        self.instant_update = False
        self.initui()
        self.binding()
        self.pdf_setting = {}

    def initui(self):
        self.setMinimumSize(800, 600)
        self.layout = QtWidgets.QHBoxLayout()
        self.controlPanel = ControlPanel(self)
        self.graphPanel = GraphPanel()

        self.layout.addWidget(self.controlPanel)
        self.layout.addWidget(self.graphPanel)
        self.layout.setStretch(1, 1)
        # self.setLayout(self.layout)
        self.show()

        self.graph_Iq = self.graphPanel.graph_Iq
        self.graph_phiq = self.graphPanel.graph_phiq
        self.graph_Gr = self.graphPanel.graph_Gr

        self.graph_Iq.addLegend(offset=(-30, 30))
        self.graph_phiq.addLegend(offset=(-30, 30))
        self.graph_Gr.addLegend(offset=(-30, 30))

        self.graph_Iq_Iq = self.graph_Iq.plot(pen=pg.mkPen(255, 0, 0, width=2), name='Iq')
        self.graph_Iq_AutoFit = self.graph_Iq.plot(pen=pg.mkPen(0, 255, 0, width=2), name='AutoFit')
        self.graph_phiq_phiq = self.graph_phiq.plot(pen=pg.mkPen(255, 0, 0, width=2), name='phiq')
        self.graph_phiq_damp = self.graph_phiq.plot(pen=pg.mkPen(0, 255, 0, width=2), name='phiq_damp')
        self.graph_Gr_Gr = self.graph_Gr.plot(pen=pg.mkPen(255, 0, 0, width=2), name='Gr')

        centralWidget = QtWidgets.QWidget()
        centralWidget.setLayout(self.layout)
        self.setCentralWidget(centralWidget)

        self.put_data_to_ui()
        if self.datacube.Gr is not None:
            self.update_graph()

    def put_data_to_ui(self):
        # elements
        print(self.datacube.element_nums)
        if self.datacube.element_nums is not None:
            for i in range(len(self.datacube.element_nums)):
                self.controlPanel.fitting_elements.element_group_widgets[i].combobox.setCurrentIndex(self.datacube.element_nums[i])
                self.controlPanel.fitting_elements.element_group_widgets[i].element_ratio.setValue(self.datacube.element_ratio[i])

        # factors
        if self.datacube.fit_at_q is not None:
            ui_util.update_value(self.controlPanel.fitting_factors.spinbox_fit_at_q,self.datacube.fit_at_q)
        if self.datacube.ds is not None:
            ui_util.update_value(self.controlPanel.fitting_factors.spinbox_ds,self.datacube.ds)
        if self.datacube.N is not None:
            ui_util.update_value(self.controlPanel.fitting_factors.spinbox_N,self.datacube.N)
        if self.datacube.damping is not None:
            ui_util.update_value(self.controlPanel.fitting_factors.spinbox_damping,self.datacube.damping)
        if self.datacube.dr is not None:
            ui_util.update_value(self.controlPanel.fitting_factors.spinbox_dr,self.datacube.dr)
        if self.datacube.is_full_q is not None:
            if self.datacube.is_full_q:
                ui_util.update_value(self.controlPanel.fitting_factors.radio_full_range,True)
            else:
                ui_util.update_value(self.controlPanel.fitting_factors.radio_tail,True)

    def update_graph(self):
        self.graph_Iq_Iq.setData(self.datacube.q, self.datacube.Iq)
        self.graph_Iq_AutoFit.setData(self.datacube.q, self.datacube.Autofit)
        self.graph_phiq_phiq.setData(self.datacube.q, self.datacube.phiq)
        self.graph_phiq_damp.setData(self.datacube.q, self.datacube.phiq_damp)
        self.graph_Gr_Gr.setData(self.datacube.r, self.datacube.Gr)

    def binding(self):
        self.controlPanel.fitting_factors.btn_auto_fit.clicked.connect(self.autofit)
        self.controlPanel.fitting_factors.btn_manual_fit.clicked.connect(self.manualfit)

        # instant fit
        self.controlPanel.fitting_factors.spinbox_N.valueChanged.connect(self.instantfit)
        self.controlPanel.fitting_factors.spinbox_dr.valueChanged.connect(self.instantfit)
        self.controlPanel.fitting_factors.spinbox_rmax.valueChanged.connect(self.instantfit)
        self.controlPanel.fitting_factors.spinbox_damping.valueChanged.connect(self.instantfit)
        self.controlPanel.fitting_factors.spinbox_fit_at_q.valueChanged.connect(self.instantfit)
        self.controlPanel.fitting_factors.spinbox_ds.valueChanged.connect(self.instantfit)
        for widget in self.controlPanel.fitting_elements.element_group_widgets:
            widget.combobox.currentIndexChanged.connect(self.instantfit)
            widget.element_ratio.valueChanged.connect(self.instantfit)

        self.controlPanel.load_and_save.load_pdf_setting.triggered.connect(self.load_pdf_setting)
        self.controlPanel.load_and_save.save_pdf_setting.triggered.connect(self.save_pdf_setting)
        self.controlPanel.load_and_save.save_pdf_setting_as.triggered.connect(self.save_pdf_setting_as)
        self.controlPanel.load_and_save.load_azavg_from_file.triggered.connect(self.load_azavg_from_file)
        self.controlPanel.load_and_save.load_azavg_from_main_window.triggered.connect(self.load_azavg_from_main_window)

    def load_pdf_setting(self):
        rs = file.load_preset_default()
        if not rs:
            rs = file.load_preset_manual()
        self.pdf_setting = rs
        self.set_pdf_setting()

    def save_pdf_setting(self):
        file.save_preset_default()
        pass

    def save_pdf_setting_as(self):
        pass

    def load_azavg_from_file(self):
        self.datacube = DataCube()
        self.datacube.azavg = file.load_azavg_manual()

    def load_azavg_from_main_window(self):
        pass

    def get_pdf_setting(self):
        for i, widget in enumerate(self.controlPanel.fitting_elements.element_group_widgets):
            self.pdf_setting.update({
                "element" + str(i): widget.combobox.currentText()
            })
            self.pdf_setting.update({
                "element_ratio" + str(i): widget.element_ratio.value()
            })
        self.pdf_setting.update({
            "Calibration_factor":   self.controlPanel.fitting_factors.spinbox_ds.value(),
            "Fit_at_Q":             self.controlPanel.fitting_factors.spinbox_fit_at_q.value(),
            "N":                    self.controlPanel.fitting_factors.spinbox_N.value(),
            "Damping":              self.controlPanel.fitting_factors.spinbox_damping.value(),
            "r(max)":               self.controlPanel.fitting_factors.spinbox_rmax.value(),
            "dr":                   self.controlPanel.fitting_factors.spinbox_dr.value()
        })
        return self.pdf_setting

    def set_pdf_setting(self):
        for i, widget in enumerate(self.controlPanel.fitting_elements.element_group_widgets):
            widget.combobox.setCurrentText(self.pdf_setting.get("element" + str(i))) # todo
            widget.element_ratio.setValue(self.pdf_setting.get("element_ratio" + str(i))) # todo
        self.controlPanel.fitting_factors.spinbox_ds.setValue(self.pdf_setting["Calibration_factor"])
        self.controlPanel.fitting_factors.spinbox_fit_at_q.setValue(self.pdf_setting["Fit_at_Q"])
        self.controlPanel.fitting_factors.spinbox_N.setValue(self.pdf_setting["N"])
        self.controlPanel.fitting_factors.spinbox_damping.setValue(self.pdf_setting["Damping"])
        self.controlPanel.fitting_factors.spinbox_rmax.setValue(self.pdf_setting["r(max)"])
        self.controlPanel.fitting_factors.spinbox_dr.setValue(self.pdf_setting["dr"])

    def update_parameter(self):
        # elements
        self.datacube.element_nums.clear()
        self.datacube.element_ratio.clear()
        for element_widget in self.controlPanel.fitting_elements.element_group_widgets:  # todo: test
            self.datacube.element_nums.append(element_widget.combobox.currentIndex())
            self.datacube.element_ratio.append(element_widget.element_ratio.value())
        self.datacube.fit_at_q = self.controlPanel.fitting_factors.spinbox_fit_at_q.value()
        self.datacube.N = self.controlPanel.fitting_factors.spinbox_N.value()
        self.datacube.damping = self.controlPanel.fitting_factors.spinbox_damping.value()
        self.datacube.rmax = self.controlPanel.fitting_factors.spinbox_rmax.value()
        self.datacube.dr = self.controlPanel.fitting_factors.spinbox_dr.value()
        self.datacube.ds = self.controlPanel.fitting_factors.spinbox_ds.value()
        self.datacube.is_full_q = self.controlPanel.fitting_factors.radio_full_range.isChecked()

    def autofit(self):
        if not self.check_condition():
            return
        self.update_parameter()
        self.datacube.q, self.datacube.r, self.datacube.Iq, self.datacube.Autofit, self.datacube.phiq, self.datacube.phiq_damp, self.datacube.Gr, self.datacube.SS, self.datacube.fit_at_q, self.datacube.N = rdf_calculator.calculation(
            self.datacube.ds,
            self.datacube.pixel_start_n,
            self.datacube.pixel_end_n,
            self.datacube.element_nums,
            self.datacube.element_ratio,
            self.datacube.azavg,
            self.datacube.is_full_q,
            self.datacube.damping,
            self.datacube.rmax,
            self.datacube.dr
        )
        print(self.datacube.N)
        ui_util.update_value(self.controlPanel.fitting_factors.spinbox_fit_at_q,self.datacube.fit_at_q)
        ui_util.update_value(self.controlPanel.fitting_factors.spinbox_N,self.datacube.N)
        # todo: add SS
        self.update_graph()

    def manualfit(self):
        if not self.check_condition():
            return
        self.update_parameter()
        self.datacube.q, self.datacube.r, self.datacube.Iq, self.datacube.Autofit, self.datacube.phiq, self.datacube.phiq_damp, self.datacube.Gr, self.datacube.SS, self.datacube.fit_at_q, self.datacube.N = rdf_calculator.calculation(
            self.datacube.ds,
            self.datacube.pixel_start_n,
            self.datacube.pixel_end_n,
            self.datacube.element_nums,
            self.datacube.element_ratio,
            self.datacube.azavg,
            self.datacube.is_full_q,
            self.datacube.damping,
            self.datacube.rmax,
            self.datacube.dr,
            self.datacube.fit_at_q,
            self.datacube.N
        )
        self.update_graph()

    def instantfit(self):
        self.update_parameter()
        if not self.controlPanel.fitting_factors.chkbox_instant_update.isChecked():
            # print("not checked")
            return
        self.datacube.q, self.datacube.r, self.datacube.Iq, self.datacube.Autofit, self.datacube.phiq, self.datacube.phiq_damp, self.datacube.Gr, self.datacube.SS, self.datacube.fit_at_q, self.datacube.N = rdf_calculator.calculation(
            self.datacube.ds,
            self.datacube.pixel_start_n,
            self.datacube.pixel_end_n,
            self.datacube.element_nums,
            self.datacube.element_ratio,
            self.datacube.azavg,
            self.datacube.is_full_q,
            self.datacube.damping,
            self.datacube.rmax,
            self.datacube.dr,
            self.datacube.fit_at_q,
            self.datacube.N
        )
        self.update_graph()

    def check_condition(self):
        if self.datacube.azavg is None:
            QMessageBox.about(self, "info", "azimuthally averaged intensity is not calculated yet.")
            return False
        return True


class GraphPanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.layout = QtWidgets.QVBoxLayout()
        # self.graph_Iq = pg.PlotWidget(title='I(q)')
        # self.graph_phiq = pg.PlotWidget(title='Φ(q)')
        # self.graph_Gr = pg.PlotWidget(title='G(r)')
        self.graph_Iq = ui_util.CoordinatesPlotWidget(title='I(q)')
        self.graph_phiq = ui_util.CoordinatesPlotWidget(title='Φ(q)')
        self.graph_Gr = ui_util.CoordinatesPlotWidget(title='G(r)')

        self.axis1 = pg.InfiniteLine(angle=0)
        self.axis2 = pg.InfiniteLine(angle=0)
        self.axis3 = pg.InfiniteLine(angle=0)

        self.graph_Iq.addItem(self.axis1)
        self.graph_phiq.addItem(self.axis2)
        self.graph_Gr.addItem(self.axis3)

        # self.layout.addWidget(self.graph_Iq)
        # self.layout.addWidget(self.graph_phiq)
        # self.layout.addWidget(self.graph_Gr)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter.addWidget(self.graph_Iq)
        self.splitter.addWidget(self.graph_phiq)
        self.splitter.addWidget(self.graph_Gr)

        self.layout.addWidget(self.splitter)
        self.setLayout(self.layout)


class ControlPanel(QtWidgets.QWidget):
    def __init__(self, mainWindow: QtWidgets.QMainWindow):
        QtWidgets.QWidget.__init__(self)
        self.layout = QtWidgets.QVBoxLayout()
        self.load_and_save = self.LoadAndSaveGroup(mainWindow)
        self.fitting_elements = self.FittingElements()
        self.fitting_factors = self.FittingFactors()

        # self.layout.addWidget(self.load_and_save)
        self.layout.addWidget(self.fitting_elements)
        self.layout.addWidget(self.fitting_factors)

        self.layout.addStretch(1)
        self.setLayout(self.layout)

    class FittingElements(QtWidgets.QGroupBox):
        def __init__(self):
            QtWidgets.QGroupBox.__init__(self)
            self.setTitle("Element")
            layout = QtWidgets.QVBoxLayout()
            layout.setSpacing(0)
            layout.setContentsMargins(10, 0, 5, 5)
            self.element_group_widgets = [ControlPanel.element_group("element" + str(num)) for num in range(1, 6)]
            for element_group_widgets in self.element_group_widgets:
                layout.addWidget(element_group_widgets)
                element_group_widgets.setContentsMargins(0, 0, 0, 0)
            self.setLayout(layout)

    class FittingFactors(QtWidgets.QGroupBox):
        def __init__(self):
            QtWidgets.QGroupBox.__init__(self)
            self.setTitle("Factors")
            layout = QtWidgets.QGridLayout()

            lbl_calibration_factor = QtWidgets.QLabel("Calibration factors")
            self.spinbox_ds = ui_util.DoubleSpinBox()
            self.spinbox_ds.setValue(0.001)
            self.spinbox_ds_step = ui_util.DoubleLineEdit()
            self.spinbox_ds_step.textChanged.connect(
                lambda : self.spinbox_ds.setSingleStep(float(self.spinbox_ds_step.text())))
            self.spinbox_ds.setRange(0,1e+10)
            self.spinbox_ds_step.setText("0.01")

            lbl_fitting_q_range = QtWidgets.QLabel("Fitting Q Range")
            self.radio_full_range = QtWidgets.QRadioButton("full range")
            self.radio_tail = QtWidgets.QRadioButton("tail")
            self.radio_full_range.setChecked(True)

            self.btn_auto_fit = QtWidgets.QPushButton("Auto Fit")

            lbl_fit_at_q = QtWidgets.QLabel("Fit at q")
            self.spinbox_fit_at_q = ui_util.DoubleSpinBox()
            self.spinbox_fit_at_q_step = ui_util.DoubleLineEdit()
            self.spinbox_fit_at_q_step.textChanged.connect(
                lambda : self.spinbox_fit_at_q.setSingleStep(float(self.spinbox_fit_at_q_step.text())))
            self.spinbox_fit_at_q.setRange(0,1e+10)
            self.spinbox_fit_at_q_step.setText("0.1")

            lbl_N = QtWidgets.QLabel("N")
            self.spinbox_N = ui_util.DoubleSpinBox()
            self.spinbox_N_step = ui_util.DoubleLineEdit()
            self.spinbox_N_step.textChanged.connect(
                lambda: self.spinbox_N.setSingleStep(float(self.spinbox_N_step.text())))
            self.spinbox_N.setRange(0, 1e+10)
            self.spinbox_N_step.setText("0.1")

            lbl_damping = QtWidgets.QLabel("Damping")
            self.spinbox_damping = ui_util.DoubleSpinBox()
            self.spinbox_damping_step = ui_util.DoubleLineEdit()
            self.spinbox_damping_step.textChanged.connect(
                lambda: self.spinbox_damping.setSingleStep(float(self.spinbox_damping_step.text())))
            self.spinbox_damping.setRange(0, 1e+10)
            self.spinbox_damping_step.setText("0.1")

            lbl_rmax = QtWidgets.QLabel("r(max)")
            self.spinbox_rmax = ui_util.DoubleSpinBox()
            self.spinbox_rmax_step = ui_util.DoubleLineEdit()
            self.spinbox_rmax_step.textChanged.connect(
                lambda: self.spinbox_rmax.setSingleStep(float(self.spinbox_rmax_step.text())))
            self.spinbox_rmax.setRange(0, 1e+10)
            self.spinbox_rmax_step.setText("1")


            lbl_dr = QtWidgets.QLabel("dr")
            self.spinbox_dr = ui_util.DoubleSpinBox()
            self.spinbox_dr_step = ui_util.DoubleLineEdit()
            self.spinbox_dr_step.textChanged.connect(
                lambda: self.spinbox_dr.setSingleStep(float(self.spinbox_dr_step.text())))
            self.spinbox_dr.setRange(0, 1e+10)
            self.spinbox_dr_step.setText("0.01")


            self.btn_manual_fit = QtWidgets.QPushButton("Manual Fit")
            layout.addWidget(self.btn_manual_fit, 8, 0, 1, 4)

            lbl_instant_update = QtWidgets.QLabel("instant update")
            self.chkbox_instant_update = QtWidgets.QCheckBox()

            self.spinbox_dr.setValue(float(util.settings["default_dr"]))
            self.spinbox_rmax.setValue(float(util.settings["default_rmax"]))
            self.spinbox_damping.setValue(float(util.settings["default_damping"]))


            layout.addWidget(lbl_calibration_factor, 0, 0)
            layout.addWidget(self.spinbox_ds, 0, 2, 1, 1)
            layout.addWidget(self.spinbox_ds_step, 0, 3, 1, 1)

            layout.addWidget(lbl_fitting_q_range, 1, 0, 1, 2)
            layout.addWidget(self.radio_full_range, 1, 2)
            layout.addWidget(self.radio_tail, 1, 3)

            layout.addWidget(self.btn_auto_fit, 2, 0, 1, 4)

            layout.addWidget(lbl_fit_at_q, 3, 0, 1, 2)
            layout.addWidget(self.spinbox_fit_at_q, 3, 2, 1, 1)
            layout.addWidget(self.spinbox_fit_at_q_step, 3, 3, 1, 1)

            layout.addWidget(lbl_N, 4, 0, 1, 2)
            layout.addWidget(self.spinbox_N, 4, 2, 1, 1)
            layout.addWidget(self.spinbox_N_step, 4, 3, 1, 1)

            layout.addWidget(lbl_damping, 5, 0, 1, 2)
            layout.addWidget(self.spinbox_damping, 5, 2, 1, 1)
            layout.addWidget(self.spinbox_damping_step, 5, 3, 1, 1)

            layout.addWidget(lbl_rmax, 6, 0, 1, 2)
            layout.addWidget(self.spinbox_rmax, 6, 2, 1, 1)
            layout.addWidget(self.spinbox_rmax_step, 6, 3, 1, 1)

            layout.addWidget(lbl_dr, 7, 0, 1, 2)
            layout.addWidget(self.spinbox_dr, 7, 2, 1, 1)
            layout.addWidget(self.spinbox_dr_step, 7, 3, 1, 1)

            layout.addWidget(lbl_instant_update, 9, 0)
            layout.addWidget(self.chkbox_instant_update, 9, 1)


            self.setLayout(layout)

    class LoadAndSaveGroup(QtWidgets.QGroupBox):
        def __init__(self, mainWindow: QtWidgets.QMainWindow):
            QtWidgets.QGroupBox.__init__(self)
            self.setTitle("Load and Save")
            layout = QtWidgets.QHBoxLayout()
            self.menu_file = self.create_menu(mainWindow)
            self.lbl_file_name = QtWidgets.QLabel("...")
            layout.addWidget(self.menu_file)
            layout.addWidget(self.lbl_file_name)
            self.setLayout(layout)

        def create_menu(self, mainWindow: QtWidgets.QMainWindow):
            menubar = mainWindow.menuBar()

            self.load_pdf_setting = QtWidgets.QAction("&Load pdf setting", self)
            self.save_pdf_setting = QtWidgets.QAction("&Save pdf setting", self)
            self.save_pdf_setting_as = QtWidgets.QAction("&Save pdf setting as", self)
            self.load_azavg_from_file = QtWidgets.QAction("&Load azavg from file", self)
            self.load_azavg_from_main_window = QtWidgets.QAction("&Load azavg from main window", self)

            filemenu = menubar.addMenu("     File     ")
            filemenu.addAction(self.load_pdf_setting)
            filemenu.addAction(self.save_pdf_setting)
            filemenu.addAction(self.save_pdf_setting_as)
            filemenu.addSeparator()
            filemenu.addAction(self.load_azavg_from_file)
            filemenu.addAction(self.load_azavg_from_main_window)

            menubar.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            return menubar

    class element_group(QtWidgets.QWidget):
        def __init__(self, label: str):
            QtWidgets.QWidget.__init__(self)
            layout = QtWidgets.QHBoxLayout()
            layout.setContentsMargins(1, 1, 1, 1)
            lbl = QtWidgets.QLabel(label)
            self.combobox = QtWidgets.QComboBox()
            self.combobox.addItems(util.get_atomic_number_symbol())
            # todo: combobox
            self.element_ratio = QtWidgets.QSpinBox()
            layout.addWidget(lbl)
            layout.addWidget(self.combobox)
            layout.addWidget(self.element_ratio)
            self.setLayout(layout)


if __name__ == "__main__":
    qtapp = QtWidgets.QApplication([])
    # QtWidgets.QMainWindow().show()
    window = rdf_analyse(DataCube("../assets/Camera 230 mm Ceta 20210312 1333_50s_20f_area01.mrc"))
    window.show()
    qtapp.exec()
