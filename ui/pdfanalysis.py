import typing

from ui.advanced_fit import AdvancedFitWindow, MainWindowAdvancedFit
import file
from datacube import DataCube
import pyqtgraph as pg
import util
from calculate import pdf_calculator
from PyQt5.QtWidgets import QMessageBox
import ui.ui_util as ui_util
from PyQt5 import QtCore, QtWidgets, QtGui
import os
import numpy as np
import definitions
from calculate import q_range_selector

pg.setConfigOptions(antialias=True)


class PdfAnalysis(QtWidgets.QWidget):
    def __init__(self, Dataviewer):
        super().__init__()
        self.Dataviewer = Dataviewer
        self.datacube = DataCube()
        self.datacube.analyser = self
        if self.datacube.element_nums is None:
            self.datacube.element_nums = []
            self.datacube.element_ratio = []
        self.instant_update = False

        self.initui()
        self.initgraph()
        self.element_presets = file.load_element_preset()

        # datacube control
        self.load_default_setting()
        self.put_data_to_ui()
        self.update_initial_iq()

        self.sig_binding()
        self.element_initial_load()

        # self.update_parameter()

        self.update_graph()

    def put_datacube(self,datacube):
        self.controlPanel.blockSignals(True)
        self.datacube = datacube
        if self.datacube.element_nums is None:
            self.datacube.element_nums = []
            self.datacube.element_ratio = []
        self.load_default_setting()
        self.put_data_to_ui()
        self.update_initial_iq()
        self.update_graph()
        self.update_initial_iq_graph()
        self.controlPanel.blockSignals(False)

    def btn_select_clicked(self):
        if hasattr(self.graph_Iq_panel.plotWidget,"select_mode") and self.graph_Iq_panel.plotWidget.select_mode is True:
            self.azav_select_exit_event()
            return
        azavg = self.datacube.azavg
        if self.datacube.azavg is None:
            return
        first_peak_idx, second_peak_idx = q_range_selector.find_multiple_peaks(self.datacube.azavg)
        self.graph_Iq_panel.plotWidget.create_circle([ui_util.pixel_to_q(first_peak_idx, self.datacube.ds), azavg[first_peak_idx]],
                                                          [ui_util.pixel_to_q(second_peak_idx, self.datacube.ds), azavg[second_peak_idx]])

        self.azav_select_enter_event()

        l = q_range_selector.find_first_nonzero_idx(self.datacube.azavg)
        l = ui_util.pixel_to_q(l, self.datacube.ds)
        r = l + int((len(self.datacube.azavg) - l) / 4)
        r = ui_util.pixel_to_q(r, self.datacube.ds)
        self.graph_Iq_panel.plotWidget.setXRange(l, r, padding=0.1)
        self.graph_Iq_panel.plotWidget.select_event = self.azav_select_exit_event

    def azav_select_exit_event(self):
        self.controlPanel.show()
        self.graph_phiq_panel.show()
        self.graph_Gr_panel.show()
        self.graph_Iq_panel.plotWidget.select_mode = False
        self.graph_Iq_panel.plotWidget.first_dev_plot.clear()
        self.graph_Iq_panel.plotWidget.first_dev_plot = None
        self.graph_Iq_panel.plotWidget.second_dev_plot.clear()
        self.graph_Iq_panel.plotWidget.second_dev_plot = None

    def azav_select_enter_event(self):
        self.controlPanel.hide()
        self.graph_phiq_panel.hide()
        self.graph_Gr_panel.hide()
        self.graph_Iq_panel.plotWidget.select_mode = True


    def update_initial_iq(self):
        if self.datacube.azavg is None:
            return
        # pixel start n
        if self.datacube.pixel_start_n is None:
            if self.datacube.q is None:
                self.datacube.pixel_start_n = q_range_selector.find_first_peak(self.datacube.azavg)
                if self.datacube.pixel_start_n is not 0:
                    self.datacube.pixel_start_n = self.datacube.pixel_start_n - 1 # why ?
                self.datacube.pixel_end_n = len(self.datacube.azavg) - 1
            else:
                self.datacube.pixel_start_n = pdf_calculator.q_to_pixel(self.datacube.q[0],self.datacube.ds)
                self.datacube.pixel_end_n = pdf_calculator.q_to_pixel(self.datacube.q[-1],self.datacube.ds)

        self.datacube.Iq = self.datacube.azavg[self.datacube.pixel_start_n:self.datacube.pixel_end_n+1]
        px = np.arange(self.datacube.pixel_start_n,self.datacube.pixel_end_n+1)
        self.datacube.q = pdf_calculator.pixel_to_q(px,self.datacube.ds)

        azavg_px = np.arange(len(self.datacube.azavg))
        self.datacube.all_q = pdf_calculator.pixel_to_q(azavg_px,self.datacube.ds)

    def update_initial_iq_graph(self):
        if self.datacube.all_q is None:
            return
        self.graph_Iq_Iq.setData(self.datacube.all_q,self.datacube.azavg)

        self.graph_Iq_panel.setting.spinBox_range_right.blockSignals(True)
        self.graph_Iq_panel.setting.spinBox_range_right.setMaximum(self.datacube.all_q[-1])
        self.graph_Iq_panel.setting.spinBox_range_left.setMaximum(self.datacube.all_q[-1])
        self.graph_Iq_panel.setting.spinBox_range_right.setSingleStep(self.datacube.ds * 2 * np.pi)
        self.graph_Iq_panel.setting.spinBox_range_left.setSingleStep(self.datacube.ds * 2 * np.pi)
        self.graph_Iq_panel.setting.spinBox_range_right.blockSignals(False)
        ui_util.update_value(self.graph_Iq_panel.region,
                             pdf_calculator.pixel_to_q([self.datacube.pixel_start_n,self.datacube.pixel_end_n],self.datacube.ds))
        self.graph_Iq_panel.range_to_dialog()

    def initui(self):
        self.controlPanel = ControlPanel(self.Dataviewer)
        self.graph_Iq_panel = GraphIqPanel()
        self.graph_phiq_panel = GraphPhiqPanel()
        self.graph_Gr_panel = GraphGrPanel()

        self.upper_left = self.controlPanel
        self.bottom_left = self.graph_Iq_panel
        self.upper_right = self.graph_phiq_panel
        self.bottom_right = self.graph_Gr_panel

        self.splitter_left_vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter_left_vertical.addWidget(self.upper_left)
        self.splitter_left_vertical.addWidget(self.bottom_left)
        self.splitter_left_vertical.setStretchFactor(1, 1)

        self.splitter_right_vertical = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter_right_vertical.addWidget(self.upper_right)
        self.splitter_right_vertical.addWidget(self.bottom_right)

        self.left = self.splitter_left_vertical
        self.right = self.splitter_right_vertical

        self.splitter_horizontal = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter_horizontal.addWidget(self.left)
        self.splitter_horizontal.addWidget(self.right)

        # window ratio
        self.splitter_horizontal.setStretchFactor(0, 8)
        self.splitter_horizontal.setStretchFactor(1, 10)

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.splitter_horizontal)

    def initgraph(self):
        self.graph_Iq = self.graph_Iq_panel.graph
        self.graph_phiq = self.graph_phiq_panel.graph
        self.graph_Gr = self.graph_Gr_panel.graph

        self.graph_Iq.addLegend(offset=(-30, 30))
        self.graph_phiq.addLegend(offset=(-30, 30))
        self.graph_Gr.addLegend(offset=(-30, 30))

        self.graph_Iq_Iq = self.graph_Iq.plot(pen=pg.mkPen(255, 0, 0, width=2), name='I')
        self.graph_Iq_AutoFit = self.graph_Iq.plot(pen=pg.mkPen(0, 255, 0, width=2), name='AutoFit')
        self.graph_phiq_phiq = self.graph_phiq.plot(pen=pg.mkPen(255, 0, 0, width=2), name='phiq')
        self.graph_phiq_damp = self.graph_phiq.plot(pen=pg.mkPen(0, 255, 0, width=2), name='phiq_damp')
        self.graph_Gr_Gr = self.graph_Gr.plot(pen=pg.mkPen(255, 0, 0, width=2), name='Gr')

        self.setLayout(self.layout)


    def load_default_setting(self):
        if util.default_setting.calibration_factor is not None and self.datacube.ds is None:
            # self.controlPanel.fitting_elements.spinbox_ds.setValue(util.default_setting.calibration_factor)
            self.datacube.ds = util.default_setting.calibration_factor
        if util.default_setting.dr is not None and self.datacube.dr is None:
            # self.controlPanel.fitting_factors.spinbox_dr.setValue(util.default_setting.dr)
            self.datacube.dr = util.default_setting.dr
        if util.default_setting.damping is not None and self.datacube.damping is None:
            # self.controlPanel.fitting_factors.spinbox_damping.setValue(util.default_setting.damping)
            self.datacube.damping = util.default_setting.damping
        if util.default_setting.rmax is not None and self.datacube.rmax is None:
            # self.controlPanel.fitting_factors.spinbox_rmax.setValue(util.default_setting.rmax)
            self.datacube.rmax = util.default_setting.rmax
        if util.default_setting.electron_voltage is not None and self.datacube.electron_voltage is None:
            # self.controlPanel.fitting_factors.spinbox_electron_voltage.setText(util.default_setting.electron_voltage)
            self.datacube.electron_voltage = util.default_setting.electron_voltage

        # steps
        if util.default_setting.calibration_factor_step is not None:
            self.controlPanel.fitting_elements.spinbox_ds_step.setText(util.default_setting.calibration_factor_step)
        if util.default_setting.fit_at_q_step is not None:
            self.controlPanel.fitting_factors.spinbox_fit_at_q_step.setText(util.default_setting.fit_at_q_step)
        if util.default_setting.N_step is not None:
            self.controlPanel.fitting_factors.spinbox_N_step.setText(util.default_setting.N_step)
        if util.default_setting.dr_step is not None:
            self.controlPanel.fitting_factors.spinbox_dr_step.setText(util.default_setting.dr_step)
        if util.default_setting.damping_step is not None:
            self.controlPanel.fitting_factors.spinbox_damping_step.setText(util.default_setting.damping_step)
        if util.default_setting.rmax_step is not None:
            self.controlPanel.fitting_factors.spinbox_rmax_step.setText(util.default_setting.rmax_step)

    def element_initial_load(self):
        self.element_presets = file.load_element_preset()
        self.controlPanel.fitting_elements.update_menu(self.element_presets)
        self.element_binding()

    def element_binding(self):
        for action in self.controlPanel.fitting_elements.load_menu.actions():
            action.triggered.connect(lambda state, x=action.text(): (self.element_load(x), self.instantfit()))
        for action in self.controlPanel.fitting_elements.save_menu.actions():
            if action.text() == "[New]":
                action.triggered.connect(lambda state: (self.element_save(), self.instantfit()))
            else:
                action.triggered.connect(lambda state, x=action.text(): (self.element_save(x), self.instantfit()))
        for action in self.controlPanel.fitting_elements.del_menu.actions():
            action.triggered.connect(lambda state, x=action.text(): (self.element_del(x), self.instantfit()))

    def element_save(self, name=None):
        # input dialog
        if name is not None:
            default_text = name
        else:
            default_text = ''
        text, ok = QtWidgets.QInputDialog.getText(self, 'Input Dialog', 'Enter preset name:', text=default_text)

        # user cancel
        if ok is False:
            return

        # check existence
        if text in self.element_presets.keys() and name is None:
            QtWidgets.QMessageBox.about(None,"Error","{} is already in the list".format(text))
            return

        # load elements data
        elements_data = {}
        for idx, widget in enumerate(self.controlPanel.fitting_elements.element_group_widgets):
            elements_data.update({"element" + str(idx+1):[widget.combobox.currentIndex(), widget.element_ratio.value()]})

        # save preset data
        new_element_presets = {}
        if name is not None:
            for k,v in self.element_presets.items():
                if name == k:
                    new_element_presets.update({text: elements_data})
                else:
                    new_element_presets.update({k: v})
            self.element_presets.clear()
            self.element_presets.update(new_element_presets)
        else:
            self.element_presets.update({text: elements_data})

        file.save_element_preset(self.element_presets)
        self.element_initial_load()

    def element_load(self, name):
        data = self.element_presets[name]
        for idx, widget in enumerate(self.controlPanel.fitting_elements.element_group_widgets):
            if "element"+str(idx+1) in data.keys():
                ui_util.update_value(widget.combobox, data["element"+str(idx+1)][0])
                ui_util.update_value(widget.element_ratio, data["element"+str(idx+1)][1])
            else:
                ui_util.update_value(widget.combobox, 0)
                ui_util.update_value(widget.element_ratio, 0)

        self.element_initial_load()

    def element_del(self, name):
        self.element_presets.pop(name)

        file.save_element_preset(self.element_presets)
        self.element_initial_load()


    def put_data_to_ui(self):
        # elements
        if self.datacube.element_nums is not None:
            # for i in range(len(self.datacube.element_nums)):
            #     self.controlPanel.fitting_elements.element_group_widgets[i].combobox.setCurrentIndex(self.datacube.element_nums[i])
            #     self.controlPanel.fitting_elements.element_group_widgets[i].element_ratio.setValue(self.datacube.element_ratio[i])
            for idx, widget in enumerate(self.controlPanel.fitting_elements.element_group_widgets):
                if idx < len(self.datacube.element_nums) and self.datacube.element_nums[idx] is not None:
                    ui_util.update_value(widget.combobox,self.datacube.element_nums[idx])
                    ui_util.update_value(widget.element_ratio,self.datacube.element_ratio[idx])
                else:
                    ui_util.update_value(widget.combobox, 0)
                    ui_util.update_value(widget.element_ratio, 0)

        # factors
        if self.datacube.fit_at_q is not None:
            ui_util.update_value(self.controlPanel.fitting_factors.spinbox_fit_at_q,self.datacube.fit_at_q)
        elif self.datacube.q is not None:
            ui_util.update_value(self.controlPanel.fitting_factors.spinbox_fit_at_q, self.datacube.q[-1])
        if self.datacube.ds is not None:
            ui_util.update_value(self.controlPanel.fitting_elements.spinbox_ds,self.datacube.ds)
        if self.datacube.N is not None:
            ui_util.update_value(self.controlPanel.fitting_factors.spinbox_N,self.datacube.N)
        if self.datacube.damping is not None:
            ui_util.update_value(self.controlPanel.fitting_factors.spinbox_damping,self.datacube.damping)
        if self.datacube.dr is not None:
            ui_util.update_value(self.controlPanel.fitting_factors.spinbox_dr,self.datacube.dr)
        if self.datacube.rmax is not None:
            ui_util.update_value(self.controlPanel.fitting_factors.spinbox_rmax, self.datacube.rmax)
        if self.datacube.is_full_q is not None:
            if self.datacube.is_full_q:
                ui_util.update_value(self.controlPanel.fitting_factors.radio_full_range,True)
            else:
                ui_util.update_value(self.controlPanel.fitting_factors.radio_tail,True)
                self.btn_radiotail_clicked()
        if self.datacube.pixel_end_n is not None:
            q_l = pdf_calculator.pixel_to_q(self.datacube.pixel_start_n,self.datacube.ds)
            q_r = pdf_calculator.pixel_to_q(self.datacube.pixel_end_n,self.datacube.ds)
            ui_util.update_value(self.graph_Iq_panel.setting.spinBox_range_left, q_l)
            ui_util.update_value(self.graph_Iq_panel.setting.spinBox_range_right, q_r)
            ui_util.update_value(self.graph_Iq_panel.region,[q_l,q_r])

    def update_graph(self):
        ######## graph I(q) ########
        self.graph_Iq_panel.datacube = self.datacube
        if self.datacube.q is not None:
            # self.graph_Iq_half_tail_Iq.setData(self.datacube.q, self.datacube.Iq)
            self.graph_Iq_Iq.setData(self.datacube.all_q, self.datacube.azavg)
            # self.graph_Iq_panel.range_to_dialog()
        else:
            self.graph_Iq_Iq.setData([0])

        if self.datacube.Autofit is not None:
            # self.graph_Iq_half_tail_AutoFit.setData(self.datacube.q, self.datacube.Autofit)
            # self.graph_Iq_half_tail.setXRange(self.datacube.q.max()/2,self.datacube.q.max())
            # self.graph_Iq_half_tail.YScaling()
            self.graph_Iq_AutoFit.setData(self.datacube.q, self.datacube.Autofit)
        else:
            self.graph_Iq_AutoFit.setData([0])

        ######## graph phi(q) ########
        if self.datacube.phiq is not None:
            self.graph_phiq_phiq.setData(self.datacube.q, self.datacube.phiq)
            self.graph_phiq_damp.setData(self.datacube.q, self.datacube.phiq_damp)
        else:
            # self.graph_phiq_phiq.clear() # i don't know why but it doens't work. It only works on the debug mode..
            self.graph_phiq_phiq.setData([0])
            self.graph_phiq_damp.setData([0])

        ######## graph G(r) ########
        if self.datacube.Gr is not None:
            self.graph_Gr_Gr.setData(self.datacube.r, self.datacube.Gr)
        else:
            self.graph_Gr_Gr.setData([0])

    def sig_binding(self):
        self.controlPanel.fitting_factors.btn_auto_fit.clicked.connect(self.autofit)
        self.controlPanel.fitting_factors.btn_advanced_fit.clicked.connect(self.advancedfit)
        self.controlPanel.fitting_factors.btn_manual_fit.clicked.connect(self.manualfit)

        # instant fit
        self.controlPanel.fitting_factors.spinbox_N.valueChanged.connect(self.instantfit)
        self.controlPanel.fitting_factors.spinbox_dr.valueChanged.connect(self.instantfit)
        self.controlPanel.fitting_factors.spinbox_rmax.valueChanged.connect(self.instantfit)
        self.controlPanel.fitting_factors.spinbox_damping.valueChanged.connect(self.instantfit)
        self.controlPanel.fitting_factors.spinbox_fit_at_q.valueChanged.connect(self.instantfit)
        self.controlPanel.fitting_elements.spinbox_ds.valueChanged.connect(self.instantfit)
        for widget in self.controlPanel.fitting_elements.element_group_widgets:
            widget.combobox.currentIndexChanged.connect(self.instantfit)
            widget.element_ratio.valueChanged.connect(self.instantfit)
        self.graph_Iq_panel.setting.spinBox_range_left.valueChanged.connect(self.instantfit)
        self.graph_Iq_panel.setting.spinBox_range_right.valueChanged.connect(self.instantfit)
        self.graph_Iq_panel.region.sigRegionChangeFinished.connect(self.instantfit)
        self.controlPanel.fitting_elements.combo_scattering_factor.currentIndexChanged.connect(self.instantfit)
        self.controlPanel.fitting_factors.spinbox_electron_voltage.textChanged.connect(self.instantfit)
        self.controlPanel.fitting_factors.radio_tail.clicked.connect(self.btn_radiotail_clicked)
        self.controlPanel.fitting_factors.radio_full_range.clicked.connect(self.btn_ratiofull_clicked)
        self.graph_Iq_panel.setting.button_select.clicked.connect(self.btn_select_clicked)
        # self.controlPanel.fitting_factors.spinbox_q_range_left.valueChanged.connect(self.fitting_q_range_changed)
        # self.controlPanel.fitting_factors.spinbox_q_range_right.valueChanged.connect(self.fitting_q_range_changed)

        self.controlPanel.fitting_elements.btn_apply_all.clicked.connect(self.btn_clicked_apply_to_all)

    def btn_clicked_apply_to_all(self):
        reply = QMessageBox.question(self,'Message',
                                               'Are you sure to apply calibration factor and element data to all?',
                                               QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.update_parameter()
            self.Dataviewer.apply_element_to_all(self.datacube)
            print("Yes")
        else:
            print("No")


    def btn_radiotail_clicked(self):
        if hasattr(self.graphPanel.graph_Iq, "region") and self.graphPanel.graph_Iq.region is not None:
            return
        self.controlPanel.fitting_factors.spinbox_q_range_left.setEnabled(True)
        self.controlPanel.fitting_factors.spinbox_q_range_right.setEnabled(True)
        ui_util.update_value(self.controlPanel.fitting_factors.spinbox_q_range_left,6.28)
        ui_util.update_value(self.controlPanel.fitting_factors.spinbox_q_range_right,self.datacube.q[-1])
        self.datacube.q_fitting_range_l = self.controlPanel.fitting_factors.spinbox_q_range_left.value()
        self.datacube.q_fitting_range_r = self.controlPanel.fitting_factors.spinbox_q_range_right.value()
        self.graphPanel.graph_Iq.region = pg.LinearRegionItem([self.datacube.q_fitting_range_l,self.datacube.q_fitting_range_r])
        self.graphPanel.graph_Iq.addItem(self.graphPanel.graph_Iq.region)

        self.graphPanel.graph_Iq.region.sigRegionChangeFinished.connect(self.range_to_dialog)
        self.controlPanel.fitting_factors.spinbox_q_range_left.valueChanged.connect(self.dialog_to_range)
        self.controlPanel.fitting_factors.spinbox_q_range_right.valueChanged.connect(self.dialog_to_range)
        self.range_fit()

    def btn_ratiofull_clicked(self):
        self.controlPanel.fitting_factors.spinbox_q_range_left.setEnabled(False)
        self.controlPanel.fitting_factors.spinbox_q_range_right.setEnabled(False)
        self.graphPanel.graph_Iq.removeItem(self.graphPanel.graph_Iq.region)
        self.graphPanel.graph_Iq.region = None
        self.autofit()

    def advancedfit(self):
        if self.datacube.azavg is None:
            QMessageBox.about(self, "", "You have to run profile extraction first.")
            return
        if self.datacube.element_nums is None:
            QMessageBox.about(self, "", "You have to put element information.")
            return
        self.advanced_fit_window = MainWindowAdvancedFit(self.datacube, self.advanced_fit_window_close_event)
        self.advanced_fit_window.show()
        pass

    def advanced_fit_window_close_event(self, idx_N, idx_qk, idx_Max_pix):
        if not self.check_condition_instant_fit():
            self.autofit()
        ui_util.update_value(self.controlPanel.fitting_factors.spinbox_N, idx_N)
        ui_util.update_value(self.controlPanel.fitting_factors.spinbox_fit_at_q, idx_qk)
        idx_Max_q = pdf_calculator.pixel_to_q(idx_Max_pix, self.datacube.ds)
        self.graph_Iq_panel.setting.spinBox_range_right.setValue(idx_Max_q)
        self.manualfit()
        pass

    def dialog_to_range(self):
        left = self.controlPanel.fitting_factors.spinbox_q_range_left.value()
        right = self.controlPanel.fitting_factors.spinbox_q_range_right.value()
        ui_util.update_value(self.graphPanel.graph_Iq.region,[left,right])
        self.datacube.q_fitting_range_l = left
        self.datacube.q_fitting_range_r = right
        self.range_fit()

    def range_to_dialog(self):
        left, right = self.graphPanel.graph_Iq.region.getRegion()
        left = np.round(left,1)
        right = np.round(right,1)
        if right > self.datacube.q[-1]:
            right = self.datacube.q[-1]
        if left < 0:
            left = 0
        ui_util.update_value(self.graphPanel.graph_Iq.region,[left, right])
        ui_util.update_value(self.controlPanel.fitting_factors.spinbox_q_range_left,left)
        ui_util.update_value(self.controlPanel.fitting_factors.spinbox_q_range_right,right)
        self.datacube.q_fitting_range_l = left
        self.datacube.q_fitting_range_r = right
        self.range_fit()

    def update_parameter(self):
        # default setting
        util.default_setting.calibration_factor = self.controlPanel.fitting_elements.spinbox_ds.value()
        util.default_setting.calibration_factor_step = self.controlPanel.fitting_elements.spinbox_ds_step.text()
        util.default_setting.electron_voltage = self.controlPanel.fitting_factors.spinbox_electron_voltage.text()
        util.default_setting.fit_at_q_step = self.controlPanel.fitting_factors.spinbox_fit_at_q_step.text()
        util.default_setting.N_step = self.controlPanel.fitting_factors.spinbox_N_step.text()
        util.default_setting.dr = self.controlPanel.fitting_factors.spinbox_dr.value()
        util.default_setting.dr_step = self.controlPanel.fitting_factors.spinbox_dr_step.text()
        util.default_setting.damping = self.controlPanel.fitting_factors.spinbox_damping.value()
        util.default_setting.damping_step = self.controlPanel.fitting_factors.spinbox_damping_step.text()
        util.default_setting.rmax = self.controlPanel.fitting_factors.spinbox_rmax.value()
        util.default_setting.rmax_step = self.controlPanel.fitting_factors.spinbox_rmax_step.text()

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
        self.datacube.ds = self.controlPanel.fitting_elements.spinbox_ds.value()
        self.datacube.is_full_q = self.controlPanel.fitting_factors.radio_full_range.isChecked()
        self.datacube.scattering_factor = self.controlPanel.fitting_elements.combo_scattering_factor.currentText()
        self.datacube.electron_voltage = self.controlPanel.fitting_factors.spinbox_electron_voltage.text()
        # fitting range parameters are reactively saved

    def autofit(self):
        if not self.check_condition():
            return
        if self.controlPanel.fitting_factors.radio_tail.isChecked():
            self.range_fit()
            return
        self.update_parameter()
        self.datacube.q, self.datacube.r, self.datacube.Iq, self.datacube.Autofit, self.datacube.phiq, self.datacube.phiq_damp, self.datacube.Gr, self.datacube.SS, self.datacube.fit_at_q, self.datacube.N = pdf_calculator.calculation(
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
            self.datacube.electron_voltage,
            scattering_factor_type=self.datacube.scattering_factor
        )
        ui_util.update_value(self.controlPanel.fitting_factors.spinbox_fit_at_q,self.datacube.fit_at_q)
        ui_util.update_value(self.controlPanel.fitting_factors.spinbox_N,self.datacube.N)
        # todo: add SS
        self.update_graph()
        if self.Dataviewer.top_menu.combo_dataQuality.currentIndex() == 0:
            self.Dataviewer.top_menu.combo_dataQuality.setCurrentIndex(1)

    def manualfit(self):
        if not self.check_condition_instant_fit():
            return
        if not self.check_condition():
            return
        self.update_parameter()
        self.datacube.q, self.datacube.r, self.datacube.Iq, self.datacube.Autofit, self.datacube.phiq, self.datacube.phiq_damp, self.datacube.Gr, self.datacube.SS, self.datacube.fit_at_q, self.datacube.N = pdf_calculator.calculation(
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
            self.datacube.electron_voltage,
            self.datacube.fit_at_q,
            self.datacube.N,
            self.datacube.scattering_factor
        )
        self.update_graph()

    def instantfit(self):
        self.update_parameter()
        if not self.controlPanel.fitting_factors.chkbox_instant_update.isChecked():
            # print("not checked")
            return
        if not self.check_condition_instant_fit():
            return
        if not self.check_condition(False):
            return
        self.datacube.q, self.datacube.r, self.datacube.Iq, self.datacube.Autofit, self.datacube.phiq, self.datacube.phiq_damp, self.datacube.Gr, self.datacube.SS, self.datacube.fit_at_q, self.datacube.N = pdf_calculator.calculation(
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
            self.datacube.electron_voltage,
            self.datacube.fit_at_q,
            self.datacube.N,
            self.datacube.scattering_factor
        )
        self.update_graph()

    def range_fit(self):
        if not self.check_condition():
            return
        self.update_parameter()
        print("range fit:",self.datacube.q_fitting_range_l,self.datacube.q_fitting_range_r)
        self.datacube.q, self.datacube.r, self.datacube.Iq, self.datacube.Autofit, self.datacube.phiq, self.datacube.phiq_damp, self.datacube.Gr, self.datacube.SS, self.datacube.fit_at_q, self.datacube.N = pdf_calculator.calculation(
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
            self.datacube.electron_voltage,
            self.datacube.fit_at_q,
            None,
            self.datacube.scattering_factor,
            [self.datacube.q_fitting_range_l,self.datacube.q_fitting_range_r]
        )
        ui_util.update_value(self.controlPanel.fitting_factors.spinbox_fit_at_q,self.datacube.fit_at_q)
        ui_util.update_value(self.controlPanel.fitting_factors.spinbox_N,self.datacube.N)
        self.update_graph()

    def check_condition(self, message:bool=True):
        if self.datacube.azavg is None:
            if message:
                QMessageBox.about(self, "info", "azimuthally averaged intensity is not calculated yet.")
            return False
        if np.array(self.datacube.element_nums).sum() == 0:
            if message:
                QMessageBox.about(self, "info", "set element first")
            return False
        if np.array(self.datacube.element_ratio).sum() == 0:
            if message:
                QMessageBox.about(self, "info", "set element ratio first")
            return False
        return True

    def check_condition_instant_fit(self):
        if self.datacube.fit_at_q == 0 or self.datacube.fit_at_q is None:
            return False
        return True


class GraphIqPanel(ui_util.ProfileGraphPanel):
    def __init__(self):
        ui_util.ProfileGraphPanel.__init__(self,"I(q)")
        self.graph = self.plotWidget
        self.axis = pg.InfiniteLine(angle=0)
        self.graph.addItem(self.axis)
        self.setting.lbl_range.setText("Q Range")


class GraphPhiqPanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.layout = QtWidgets.QVBoxLayout()
        self.graph = ui_util.CoordinatesPlotWidget(title='Φ(q)')
        self.axis = pg.InfiniteLine(angle=0)
        self.graph.addItem(self.axis)
        self.layout.addWidget(self.graph)
        self.layout.setContentsMargins(5,5,5,5)
        self.setLayout(self.layout)


class GraphGrPanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.layout = QtWidgets.QVBoxLayout()
        self.graph = ui_util.CoordinatesPlotWidget(title='G(r)')
        self.axis = pg.InfiniteLine(angle=0)
        self.graph.addItem(self.axis)
        self.layout.addWidget(self.graph)
        self.layout.setContentsMargins(5,5,5,5)
        self.setLayout(self.layout)


class ControlPanel(QtWidgets.QWidget):
    def __init__(self, mainWindow: QtWidgets.QMainWindow):
        QtWidgets.QWidget.__init__(self)
        self.layout = QtWidgets.QHBoxLayout()
        self.fitting_elements = self.FittingElements(mainWindow)
        self.fitting_factors = self.FittingFactors()

        self.layout.addWidget(self.fitting_elements)
        self.layout.addWidget(self.fitting_factors)

        # self.resize(600,1000)
        self.setLayout(self.layout)
        self.layout.setContentsMargins(2,2,2,2)


    class FittingElements(QtWidgets.QGroupBox):
        def __init__(self, mainWindow:QtWidgets.QMainWindow):
            QtWidgets.QGroupBox.__init__(self)
            self.setTitle("Element")
            layout = QtWidgets.QVBoxLayout()
            layout.setSpacing(0)
            # layout.setContentsMargins(10, 0, 5, 5)
            menubar = self.create_menu(mainWindow)
            layout.addWidget(menubar,alignment=QtCore.Qt.AlignCenter)



            self.element_group_widgets = [ControlPanel.element_group("Element" + str(num)) for num in range(1, 6)]
            for element_group_widgets in self.element_group_widgets:
                layout.addWidget(element_group_widgets)
            layout.addWidget(ui_util.QHLine())
            layout.addWidget(self.scattering_factors_widget())

            lbl_calibration_factor = QtWidgets.QLabel("Calibration factors")


            self.spinbox_ds = ui_util.DoubleSpinBox()
            self.spinbox_ds.setValue(0.001)
            self.spinbox_ds_step = ui_util.DoubleLineEdit()
            self.spinbox_ds_step.textChanged.connect(
                lambda : self.spinbox_ds.setSingleStep(float(self.spinbox_ds_step.text())))
            self.spinbox_ds.setRange(0,1e+10)
            self.spinbox_ds_step.setText("0.01")

            layout_calibration_factor = QtWidgets.QHBoxLayout()
            layout_calibration_factor.addWidget(lbl_calibration_factor)
            layout_calibration_factor.addWidget(self.spinbox_ds)
            layout_calibration_factor.addWidget(self.spinbox_ds_step)

            layout.addLayout(layout_calibration_factor)
            layout.addWidget(ui_util.QHLine())

            self.btn_apply_all = QtWidgets.QPushButton("Apply to all")
            layout.addWidget(self.btn_apply_all)



            self.setLayout(layout)

        def scattering_factors_widget(self):
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout()
            layout.setSpacing(0)
            layout.setContentsMargins(2,2,2,2)
            widget.setLayout(layout)
            self.lbl_scattering_factor = QtWidgets.QLabel("Scattering Factor")
            layout.addWidget(self.lbl_scattering_factor)
            self.combo_scattering_factor = QtWidgets.QComboBox()
            self.combo_scattering_factor.addItems(["Kirkland","Lobato"])
            layout.addWidget(self.combo_scattering_factor)
            return widget


        def create_menu(self, mainWindow: QtWidgets.QMainWindow):
            menubar = mainWindow.menuBar()
            menubar.setNativeMenuBar(False)
            # menu_frame_widget_layout.setSpacing(0)

            self.load_menu = menubar.addMenu("  &Load  ")
            self.save_menu = menubar.addMenu("  &Save  ")
            self.actions_new_preset = QtWidgets.QAction("[New]", self)
            self.save_menu.addAction(self.actions_new_preset)
            self.del_menu = menubar.addMenu("  &Del  ")

            menubar.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

            return menubar

        def update_menu(self, data: dict):
            # clear menu
            self.load_menu.clear()
            self.save_menu.clear()
            self.del_menu.clear()
            # for action in self.load_menu.actions():
            #     self.load_menu.removeAction(action)
            # for action in self.save_menu.actions():
            #     self.save_menu.removeAction(action)
            # for action in self.del_menu.actions():
            #     self.del_menu.removeAction(action)

            # add menu
            for k,v in data.items():
                action_load = QtWidgets.QAction(k, self)
                self.load_menu.addAction(action_load)
                # self.actions_load_preset.append(action_load)

                action_save = QtWidgets.QAction(k, self)
                self.save_menu.addAction(action_save)
                # self.actions_save_preset.append(action_save)

                action_del = QtWidgets.QAction(k, self)
                self.del_menu.addAction(action_del)
                # self.actions_del_preset.append(action_del)
            ## add new in save menu
            self.actions_new_preset = QtWidgets.QAction("[New]", self)
            self.save_menu.addAction(self.actions_new_preset)
            # self.save_menu.addAction(self.actions_new_preset)


    class FittingFactors(QtWidgets.QGroupBox):
        def __init__(self):
            QtWidgets.QGroupBox.__init__(self)
            self.setTitle("Factors")
            layout = QtWidgets.QGridLayout()

            lbl_fitting_q_range = QtWidgets.QLabel("Fitting Q Range")
            self.radio_full_range = QtWidgets.QRadioButton("full range")
            self.radio_tail = QtWidgets.QRadioButton("select")
            self.radio_full_range.setChecked(True)
            ########### Temporary disable ##############
            lbl_fitting_q_range.setDisabled(True)
            self.radio_full_range.setDisabled(True)
            self.radio_tail.setDisabled(True)
            #############################################

            autofit_button_layout = QtWidgets.QHBoxLayout()
            self.btn_auto_fit = QtWidgets.QPushButton("Auto fitting")
            self.btn_advanced_fit = QtWidgets.QPushButton("Advanced fitting")
            autofit_button_layout.addWidget(self.btn_auto_fit)
            autofit_button_layout.addWidget(self.btn_advanced_fit)
            # self.btn_auto_fit.setMaximumWidth(30)
            # self.btn_auto_fit.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,QtWidgets.QSizePolicy.Policy.Expanding)

            lbl_fit_at_q = QtWidgets.QLabel("Fit at q")
            self.spinbox_fit_at_q = QtWidgets.QDoubleSpinBox()
            self.spinbox_fit_at_q.setDecimals(3)
            self.spinbox_fit_at_q_step = ui_util.DoubleLineEdit()
            self.spinbox_fit_at_q_step.textChanged.connect(
                lambda : self.spinbox_fit_at_q.setSingleStep(float(self.spinbox_fit_at_q_step.text())))
            self.spinbox_fit_at_q.setRange(0,1e+10)
            self.spinbox_fit_at_q_step.setText("0.1")

            self.spinbox_q_range_left = ui_util.DoubleSpinBox()
            self.spinbox_q_range_left.setSingleStep(0.1)
            self.spinbox_q_range_left.setEnabled(False)
            self.spinbox_q_range_right = ui_util.DoubleSpinBox()
            self.spinbox_q_range_right.setSingleStep(0.1)
            self.spinbox_q_range_right.setEnabled(False)


            lbl_N = QtWidgets.QLabel("N")
            self.spinbox_N = QtWidgets.QDoubleSpinBox()
            self.spinbox_N.setDecimals(3)
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

            layout_last_line = QtWidgets.QHBoxLayout()
            lbl_electron_voltage = QtWidgets.QLabel("EV / kW")
            self.spinbox_electron_voltage = ui_util.DoubleLineEdit()
            self.spinbox_electron_voltage.setMaximumWidth(50)
            self.spinbox_electron_voltage.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,QtWidgets.QSizePolicy.Policy.Fixed)
            lbl_instant_update = QtWidgets.QLabel("Instant update")
            self.chkbox_instant_update = QtWidgets.QCheckBox()
            layout_last_line.addWidget(lbl_instant_update)
            layout_last_line.addWidget(self.chkbox_instant_update)
            layout_last_line.addWidget(lbl_electron_voltage)
            layout_last_line.addWidget(self.spinbox_electron_voltage)
            # self.btn_manual_fit.setMaximumWidth(30)
            # self.btn_manual_fit.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
            self.btn_manual_fit = QtWidgets.QPushButton("Manual fitting")

            # layout.addWidget(lbl_calibration_factor, 0, 0)
            # layout.addWidget(self.spinbox_ds, 0, 2, 1, 1)
            # layout.addWidget(self.spinbox_ds_step, 0, 3, 1, 1)

            layout.addWidget(lbl_fitting_q_range, 0, 0, 1, 2)
            layout.addWidget(self.radio_full_range, 0, 2)
            layout.addWidget(self.radio_tail, 0, 3)
            layout.addWidget(self.spinbox_q_range_left, 1, 2, 1, 1)
            layout.addWidget(self.spinbox_q_range_right, 1, 3, 1, 1)

            layout.addLayout(autofit_button_layout,2,0,1,5)
            # layout.addWidget(self.btn_auto_fit, 2, 0,1,2)
            # layout.addWidget(self.btn_advanced_fit, 2, 2, 1, 3)

            layout.addWidget(ui_util.QHLine(),3,0,1,5)

            layout.addWidget(lbl_fit_at_q, 4, 0, 1, 2)
            layout.addWidget(self.spinbox_fit_at_q, 4, 2, 1, 1)
            layout.addWidget(self.spinbox_fit_at_q_step, 4, 3, 1, 1)

            layout.addWidget(lbl_N, 5, 0, 1, 2)
            layout.addWidget(self.spinbox_N, 5, 2, 1, 1)
            layout.addWidget(self.spinbox_N_step, 5, 3, 1, 1)

            layout.addWidget(lbl_damping, 6, 0, 1, 2)
            layout.addWidget(self.spinbox_damping, 6, 2, 1, 1)
            layout.addWidget(self.spinbox_damping_step, 6, 3, 1, 1)

            layout.addWidget(lbl_rmax, 7, 0, 1, 2)
            layout.addWidget(self.spinbox_rmax, 7, 2, 1, 1)
            layout.addWidget(self.spinbox_rmax_step, 7, 3, 1, 1)

            layout.addWidget(lbl_dr, 8, 0, 1, 2)
            layout.addWidget(self.spinbox_dr, 8, 2, 1, 1)
            layout.addWidget(self.spinbox_dr_step, 8, 3, 1, 1)

            layout.addLayout(layout_last_line,9,0,1,5)

            # layout.addWidget(ui_util.QHLine(), 10, 0, 1, 5)
            layout.addWidget(self.btn_manual_fit, 10, 0, 1, 5)
            # layout.addWidget(lbl_instant_update, 10, 0, 1, 2)

            layout.setSpacing(1)
            # layout.setContentsMargins(0,0,0,0)

            self.setLayout(layout)

    class element_group(QtWidgets.QWidget):
        def __init__(self, label: str):
            QtWidgets.QWidget.__init__(self)
            layout = QtWidgets.QHBoxLayout()
            layout.setContentsMargins(0,0,0,0)
            layout.setSpacing(0)
            lbl = QtWidgets.QLabel(label)
            self.combobox = QtWidgets.QComboBox()
            self.combobox.addItems(util.get_atomic_number_symbol())
            # todo: combobox
            self.element_ratio = QtWidgets.QSpinBox()
            self.element_ratio.setMaximum(10000000)
            layout.addWidget(lbl)
            layout.addWidget(self.combobox)
            layout.addWidget(self.element_ratio)
            self.setLayout(layout)


if __name__ == "__main__":
    qtapp = QtWidgets.QApplication([])
    # QtWidgets.QMainWindow().show()
    window = PdfAnalysis(DataCube())
    window.show()
    qtapp.exec()
