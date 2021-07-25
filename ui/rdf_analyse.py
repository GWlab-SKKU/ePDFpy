from datacube import DataCube
import pyqtgraph as pg
import util
from calculate import rdf_calculator
from PyQt5.QtWidgets import QMessageBox

from PyQt5 import QtCore, QtWidgets


class rdf_analyse(QtWidgets.QWidget):
    def __init__(self, datacube: DataCube):
        QtWidgets.QWidget.__init__(self)
        print("init")
        self.datacube = datacube
        self.element_nums = []
        self.ratio = []
        self.instant_update = False
        self.initui()
        self.binding()
        self.datacube.analyser = self


    def initui(self):
        self.setMinimumSize(500, 500)
        self.layout = QtWidgets.QHBoxLayout()
        self.controlPanel = ControlPanel()
        self.graphPanel = GraphPanel()

        self.layout.addWidget(self.controlPanel)
        self.layout.addWidget(self.graphPanel)
        self.layout.setStretch(1, 1)
        self.setLayout(self.layout)
        self.show()

        self.graph_Iq = self.graphPanel.graph_Iq
        self.graph_phiq = self.graphPanel.graph_phiq
        self.graph_Gr = self.graphPanel.graph_Gr

        self.graph_Iq.addLegend(offset=(-30, 30))
        self.graph_phiq.addLegend(offset=(-30, 30))
        self.graph_Gr.addLegend(offset=(-30, 30))

        self.graph_Iq_Iq = self.graph_Iq.plot(pen=pg.mkPen(255, 0, 0, width=2), name='Iq')
        self.graph_Iq_AutoFit = self.graph_Iq.plot( pen=pg.mkPen(0, 255, 0, width=2), name='AutoFit')
        self.graph_phiq_phiq = self.graph_phiq.plot(pen=pg.mkPen(255, 0, 0, width=2), name='phiq')
        self.graph_phiq_damp = self.graph_phiq.plot(pen=pg.mkPen(0, 255, 0, width=2), name='phiq_damp')
        self.graph_Gr_Gr = self.graph_Gr.plot(pen=pg.mkPen(255, 0, 0, width=2), name='Gr')

    def update_graph(self):
        self.graph_Iq_Iq.setData(self.q, self.Iq)
        self.graph_Iq_AutoFit.setData(self.q, self.Autofit)
        self.graph_phiq_phiq.setData(self.q, self.phiq)
        self.graph_phiq_damp.setData(self.q, self.phiq_damp)
        self.graph_Gr_Gr.setData(self.r, self.Gr)

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
            widget.ratio.valueChanged.connect(self.instantfit)


    def update_parameter(self):
        # elements
        for element_widget in self.controlPanel.fitting_elements.element_group_widgets: # todo: test
            self.element_nums.append(element_widget.combobox.currentIndex())
            self.ratio.append(element_widget.ratio.value())
        self.fit_at_q = self.controlPanel.fitting_factors.spinbox_fit_at_q.value()
        self.N = self.controlPanel.fitting_factors.spinbox_N.value()
        self.damping = self.controlPanel.fitting_factors.spinbox_damping.value()
        self.rmax = self.controlPanel.fitting_factors.spinbox_rmax.value()
        self.dr = self.controlPanel.fitting_factors.spinbox_dr.value()
        self.datacube.ds = self.controlPanel.fitting_factors.spinbox_ds.value()
        self.is_full_q = self.controlPanel.fitting_factors.radio_full_range.isChecked()

    def autofit(self):
        if not self.check_condition():
            return
        self.update_parameter()
        self.q, self.r, self.Iq, self.Autofit, self.phiq, self.phiq_damp, self.Gr, self.SS, self.fit_at_q, self.N = rdf_calculator.calculation(
            self.datacube.ds,
            self.datacube.q_start_num,
            self.datacube.q_end_num,
            self.element_nums,
            self.ratio,
            self.datacube.azavg,
            self.is_full_q,
            self.damping,
            self.rmax,
            self.dr
        )

        self.controlPanel.fitting_factors.spinbox_fit_at_q.setValue(self.fit_at_q)
        self.controlPanel.fitting_factors.spinbox_N.setValue(self.N)
        # todo: add SS
        self.update_graph()

    def manualfit(self):
        if not self.check_condition():
            return
        self.update_parameter()
        self.q, self.r, self.Iq, self.Autofit, self.phiq, self.phiq_damp, self.Gr, self.SS, self.fit_at_q, self.N = rdf_calculator.calculation(
            self.datacube.ds,
            self.datacube.q_start_num,
            self.datacube.q_end_num,
            self.element_nums,
            self.ratio,
            self.datacube.azavg,
            self.is_full_q,
            self.damping,
            self.rmax,
            self.dr,
            self.fit_at_q,
            self.N
        )
        self.update_graph()

    def instantfit(self):
        if not self.controlPanel.fitting_factors.chkbox_instant_update.isChecked():
            print("not checked")
            return
        self.update_parameter()
        self.q, self.r, self.Iq, self.Autofit, self.phiq, self.phiq_damp, self.Gr, self.SS, self.fit_at_q, self.N = rdf_calculator.calculation(
            self.datacube.ds,
            self.datacube.q_start_num,
            self.datacube.q_end_num,
            self.element_nums,
            self.ratio,
            self.datacube.azavg,
            self.is_full_q,
            self.damping,
            self.rmax,
            self.dr,
            self.fit_at_q,
            self.N
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
        self.graph_Iq = pg.PlotWidget(title='I(q)')
        self.graph_phiq = pg.PlotWidget(title='Φ(q)')
        self.graph_Gr = pg.PlotWidget(title='G(r)')

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
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.layout = QtWidgets.QVBoxLayout()
        self.fitting_elements = self.FittingElements()
        self.fitting_factors = self.FittingFactors()
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
            layout.setContentsMargins(0,0,0,0)
            self.element_group_widgets = [ControlPanel.element_group("element" + str(num)) for num in range(1, 6)]
            for element_group_widgets in self.element_group_widgets:
                layout.addWidget(element_group_widgets)
                element_group_widgets.setContentsMargins(0,0,0,0)
            self.setLayout(layout)

    class FittingFactors(QtWidgets.QGroupBox):
        def __init__(self):
            QtWidgets.QGroupBox.__init__(self)
            self.setTitle("Factors")
            layout = QtWidgets.QGridLayout()

            lbl_calibration_factor = QtWidgets.QLabel("Calibration factors")
            self.spinbox_ds = QtWidgets.QDoubleSpinBox()
            layout.addWidget(lbl_calibration_factor,0,0)
            layout.addWidget(self.spinbox_ds,0,3)
            self.spinbox_ds.setDecimals(5)
            self.spinbox_ds.setValue(0.001)
            self.spinbox_ds.setSingleStep(0.001)
            self.spinbox_ds.setMaximum(1)


            lbl_fitting_q_range = QtWidgets.QLabel("Fitting Q Range")
            self.radio_full_range = QtWidgets.QRadioButton("full range")
            self.radio_tail = QtWidgets.QRadioButton("tail")
            layout.addWidget(lbl_fitting_q_range,1,0,1,2)
            layout.addWidget(self.radio_full_range, 1,2)
            layout.addWidget(self.radio_tail,1,3)
            self.radio_full_range.setChecked(True)

            self.btn_auto_fit = QtWidgets.QPushButton("Auto Fit")
            layout.addWidget(self.btn_auto_fit,2,0,1,4)

            lbl_fit_at_q = QtWidgets.QLabel("Fit at q")
            self.spinbox_fit_at_q = QtWidgets.QDoubleSpinBox()
            self.spinbox_fit_at_q.setMaximum(100000)
            layout.addWidget(lbl_fit_at_q,3,0,1,2)
            layout.addWidget(self.spinbox_fit_at_q, 3, 2, 1, 2)
            lbl_N = QtWidgets.QLabel("N")
            self.spinbox_N = QtWidgets.QDoubleSpinBox()
            self.spinbox_N.setMaximum(100000)
            layout.addWidget(lbl_N,4,0,1,2)
            layout.addWidget(self.spinbox_N, 4, 2, 1, 2)
            lbl_damping = QtWidgets.QLabel("Damping")
            self.spinbox_damping = QtWidgets.QDoubleSpinBox()
            layout.addWidget(lbl_damping,5,0,1,2)
            layout.addWidget(self.spinbox_damping, 5, 2, 1, 2)
            lbl_rmax = QtWidgets.QLabel("r(max)")
            self.spinbox_rmax = QtWidgets.QDoubleSpinBox()
            layout.addWidget(lbl_rmax,6,0,1,2)
            layout.addWidget(self.spinbox_rmax, 6, 2, 1, 2)
            lbl_dr = QtWidgets.QLabel("dr")
            self.spinbox_dr = QtWidgets.QDoubleSpinBox()
            layout.addWidget(lbl_dr,7,0,1,2)
            layout.addWidget(self.spinbox_dr, 7, 2, 1, 2)
            self.btn_manual_fit = QtWidgets.QPushButton("Manual Fit")
            layout.addWidget(self.btn_manual_fit,8,0,1,4)

            lbl_instant_update = QtWidgets.QLabel("instant update")
            self.chkbox_instant_update = QtWidgets.QCheckBox()
            layout.addWidget(lbl_instant_update, 9,0)
            layout.addWidget(self.chkbox_instant_update, 9, 1)

            self.spinbox_dr.setValue(float(util.settings["default_dr"]))
            self.spinbox_rmax.setValue(float(util.settings["default_rmax"]))
            self.spinbox_damping.setValue(float(util.settings["default_damping"]))

            self.setLayout(layout)

    class element_group(QtWidgets.QWidget):
        def __init__(self, label:str):
            QtWidgets.QWidget.__init__(self)
            layout = QtWidgets.QHBoxLayout()
            layout.setContentsMargins(1,1,1,1)
            lbl = QtWidgets.QLabel(label)
            self.combobox = QtWidgets.QComboBox()
            self.combobox.addItems(util.get_atomic_number_symbol())
            # todo: combobox
            self.ratio = QtWidgets.QSpinBox()
            layout.addWidget(lbl)
            layout.addWidget(self.combobox)
            layout.addWidget(self.ratio)
            self.setLayout(layout)



if __name__ == "__main__":
    qtapp = QtWidgets.QApplication([])
    # QtWidgets.QMainWindow().show()
    window = rdf_analyse(DataCube("../assets/Camera 230 mm Ceta 20210312 1333_50s_20f_area01.mrc"))
    window.show()
    qtapp.exec()