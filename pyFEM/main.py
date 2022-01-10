from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import pyqtgraph as pg
import numpy as np
import pandas as pd
from calculate import image_process, polar_transform
from ui import ui_util
from ui.roi_selector import roi_selector
import definitions
import os
import hyperspy.api as hs
import h5py
import cv2

class MainViewer(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.centralWidget = QWidget()

        self.top_menu = TopMenu(self)
        self.body = QWidget()

        self.centralWidget.layout = QHBoxLayout()
        self.centralWidget.layout.addWidget(self.top_menu)
        self.centralWidget.layout.addWidget(self.body)
        self.centralWidget.setLayout(self.centralWidget.layout)
        self.raw_img = None

        self.setCentralWidget(self.centralWidget)
        self.body_init_ui()
        self.binding()

    def body_init_ui(self):
        self.img_panel = ImgPanel()
        self.control_panel = ControlPanel(self)
        self.profile_graph_panel = IntensityProfilePanel()
        self.polar_image_panel = PolarImagePanel()

        self.upper_left = self.control_panel
        self.bottom_left = self.img_panel
        self.upper_right = self.profile_graph_panel
        self.bottom_right = self.polar_image_panel

        self.splitter_left_vertical = QSplitter(QtCore.Qt.Vertical)
        self.splitter_left_vertical.addWidget(self.upper_left)
        self.splitter_left_vertical.addWidget(self.bottom_left)
        self.splitter_left_vertical.setStretchFactor(1, 1)

        self.splitter_right_vertical = QSplitter(QtCore.Qt.Vertical)
        self.splitter_right_vertical.addWidget(self.upper_right)
        self.splitter_right_vertical.addWidget(self.bottom_right)

        self.left = self.splitter_left_vertical
        self.right = self.splitter_right_vertical

        self.splitter_horizontal = QSplitter(QtCore.Qt.Horizontal)
        self.splitter_horizontal.addWidget(self.left)
        self.splitter_horizontal.addWidget(self.right)
        self.splitter_horizontal.setStretchFactor(0, 10)
        self.splitter_horizontal.setStretchFactor(1, 7)

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(3, 3, 3, 3)
        self.layout.addWidget(self.splitter_horizontal)

        self.body.setLayout(self.layout)

    def file_load(self):
        path, _ = QFileDialog.getOpenFileName(self,'open', filter="")
        if len(path) == 0:
            return
        if ".h5" in os.path.splitext(path)[1]:
            f = h5py.File(r"/mnt/experiment/TEM diffraction/211209/211209_Si_100ms_128x128/211209_Si_100ms_128x128.h5",
                          'r')
            dc1 = np.array(f['4DSTEM_experiment']['data']['datacubes']['datacube_0']['data'])
            self.dc = dc1.reshape([-1, dc1.shape[0], dc1.shape[1]])
        elif ".dm3" in os.path.splitext(path)[1]:
            self.dc = hs.load(path).data
        else:
            return

        ui_util.update_value(self.control_panel.grp_ellipse.spinbox_x,self.dc.shape[1] // 2)
        ui_util.update_value(self.control_panel.grp_ellipse.spinbox_y, self.dc.shape[2] // 2)
        ui_util.update_value(self.control_panel.grp_ellipse.spinbox_a, 1)
        ui_util.update_value(self.control_panel.grp_ellipse.spinbox_b, 1)
        ui_util.update_value(self.control_panel.grp_ellipse.spinbox_phi, 0)

        # reset
        self.update_raw_img()
        self.update_img()

    def binding(self):
        self.control_panel.grp_view.grp_img_view.buttonToggled.connect(self.update_img)
        self.control_panel.grp_view.grp_img_integration.buttonToggled.connect(lambda: (self.update_raw_img, self.update_img))
        self.control_panel.grp_mask.cmb_mask.currentIndexChanged.connect(self.update_img)
        self.control_panel.grp_mask.chk_mask_inverse.toggled.connect(self.update_img)
        self.top_menu.open_img_file.triggered.connect(self.file_load)

    def update_raw_img(self):
        self.img_integration_mode = self.control_panel.grp_view.grp_img_integration.checkedButton().text()
        if self.img_integration_mode == "Mean":
            # mean
            self.raw_img = np.mean(self.dc, axis=0)
        elif self.img_integration_mode == "Std":
            # std
            self.raw_img = np.std(self.dc, axis=0)

    def mask_update(self):
        self.mask_data = self.control_panel.grp_mask.cmb_mask.get_current_mask()
        if self.dc is None or\
            self.mask_data is None or\
            self.control_panel.grp_mask.cmb_mask.currentIndex() == 0:
            return
        if self.mask_data.shape != self.dc.shape[1:]:
            QMessageBox.about(self,"","mask is not matching the image")
            self.control_panel.grp_mask.cmb_mask.setCurrentIndex(0)
            self.mask_data = None
        if self.control_panel.grp_mask.chk_mask_inverse.isChecked():
            self.mask_data = ~self.mask_data
        self.control_panel.grp_mask.cmb_mask.image = self.display_img


    def update_img(self):
        self.img_view_mode = self.control_panel.grp_view.grp_img_view.checkedButton().text()
        if self.img_view_mode == "Raw":
            self.display_img = self.raw_img
        elif self.img_view_mode == "Root":
            # root
            self.display_img = np.power(self.raw_img,0.5)
        elif self.img_view_mode == "Log":
            # log
            if self.display_img.min() < 1:
                self.display_img = np.abs(self.raw_img)+1
            self.display_img = np.log(self.display_img)



        self.mask_update()

        # polar transformation
        x = self.control_panel.grp_ellipse.spinbox_x.value()
        y = self.control_panel.grp_ellipse.spinbox_y.value()
        a = self.control_panel.grp_ellipse.spinbox_a.value()
        b = self.control_panel.grp_ellipse.spinbox_b.value()
        phi = self.control_panel.grp_ellipse.spinbox_phi.value()
        p_ellipse = [x,y,a,b,phi]

        self.polar_img = polar_transform.cartesian_to_polarelliptical_transform(
            cartesianData=self.display_img,
            p_ellipse=p_ellipse,
            dphi=np.radians(1),
            mask=self.mask_data)

        self.img_panel.update_img(self.display_img, self.mask_data)
        self.polar_image_panel.update_img(self.polar_img[0])





class ControlPanel(QWidget):
    def __init__(self, mainWindow):
        super().__init__()
        self.grp_mask = self.GroupMask(mainWindow.img_panel.imageView)
        self.grp_ellipse = self.GroupEllipse()
        self.grp_view = self.GroupView()
        self.grp_operation = self.GroupOperator()

        # self.layout = QGridLayout()
        self.layout = QHBoxLayout()
        self.layout_left = QVBoxLayout()
        self.layout_left.addWidget(self.grp_mask)
        self.layout_left.addWidget(self.grp_view)
        self.layout_left.addWidget(self.grp_operation)
        self.layout.addLayout(self.layout_left)
        self.layout.addWidget(self.grp_ellipse)
        self.setLayout(self.layout)

    class GroupEllipse(QGroupBox):
        def __init__(self):
            super().__init__()
            self.setTitle("Elliptical Fitting")
            layout = QGridLayout()
            self.setLayout(layout)
            self.btn_autofit = QPushButton("Auto fit")
            self.btn_manualfit = QPushButton("Manual fit")
            lbl_ellipse_x = QLabel("x")
            lbl_ellipse_y = QLabel("y")
            lbl_ellipse_a = QLabel("a")
            lbl_ellipse_b = QLabel("b")
            lbl_ellipse_phi = QLabel("phi")
            self.spinbox_x = QDoubleSpinBox()
            self.spinbox_x.setMaximum(10e5)
            self.spinbox_y = QDoubleSpinBox()
            self.spinbox_y.setMaximum(10e5)
            self.spinbox_a = QDoubleSpinBox()
            self.spinbox_b = QDoubleSpinBox()
            self.spinbox_phi = QDoubleSpinBox()
            layout.addWidget(self.btn_autofit,0,0,1,2)
            layout.addWidget(lbl_ellipse_x, 1, 0)
            layout.addWidget(lbl_ellipse_y, 2, 0)
            layout.addWidget(lbl_ellipse_a, 3, 0)
            layout.addWidget(lbl_ellipse_b, 4, 0)
            layout.addWidget(lbl_ellipse_phi, 5, 0)
            layout.addWidget(self.spinbox_x, 1, 1)
            layout.addWidget(self.spinbox_y, 2, 1)
            layout.addWidget(self.spinbox_a, 3, 1)
            layout.addWidget(self.spinbox_b, 4, 1)
            layout.addWidget(self.spinbox_phi, 5, 1)
            layout.addWidget(self.btn_manualfit, 6, 0, 1, 2)

    class GroupMask(QGroupBox):
        def __init__(self, imageView):
            super().__init__()
            self.setTitle("Mask")
            layout = QVBoxLayout()
            self.cmb_mask = roi_selector.MaskDropdown(imageView, mask_folder=definitions.MASK_FOLDER_PATH)
            self.chk_mask_inverse = QtWidgets.QCheckBox("inverse")
            layout.addWidget(self.cmb_mask)
            layout.addWidget(self.chk_mask_inverse)
            self.setLayout(layout)
            pass

    class GroupView(QGroupBox):
        def __init__(self):
            super().__init__()
            self.setTitle("View Mode")
            layout = QVBoxLayout()
            self.setLayout(layout)

            self.grp_img_integration = QButtonGroup()
            self.radio_integration_mean = QRadioButton("Mean")
            self.radio_integration_std = QRadioButton("Std")
            self.grp_img_integration.addButton(self.radio_integration_mean)
            self.grp_img_integration.addButton(self.radio_integration_std)
            layout_integration_mode = QHBoxLayout()
            layout_integration_mode.addWidget(self.radio_integration_mean)
            layout_integration_mode.addWidget(self.radio_integration_std)

            self.grp_img_view = QButtonGroup()
            self.radio_view_raw = QRadioButton("Raw")
            self.radio_view_root = QRadioButton("Root")
            self.radio_view_log = QRadioButton("Log")
            self.grp_img_view.addButton(self.radio_view_raw)
            self.grp_img_view.addButton(self.radio_view_root)
            self.grp_img_view.addButton(self.radio_view_log)
            layout_view_mode = QHBoxLayout()
            layout_view_mode.addWidget(self.radio_view_raw)
            layout_view_mode.addWidget(self.radio_view_root)
            layout_view_mode.addWidget(self.radio_view_log)

            layout.addLayout(layout_integration_mode)
            layout.addWidget(ui_util.QHLine())
            layout.addLayout(layout_view_mode)

            self.radio_view_raw.setChecked(True)
            self.radio_integration_mean.setChecked(True)

    class GroupOperator(QGroupBox):
        def __init__(self):
            super().__init__()
            self.setTitle("Operator")
            self.layout = QHBoxLayout()
            self.setLayout(self.layout)

            self.btn_fem = QtWidgets.QPushButton("run FEM")
            self.layout.addWidget(self.btn_fem)




class TopMenu(QWidget):
    def __init__(self, mainWindow):
        super().__init__()
        self.mainWindow = mainWindow
        self.open_img_file = QtWidgets.QAction("Open &image file", self)

        menubar = mainWindow.menuBar()
        file_menu = menubar.addMenu("\t&File\t")
        file_menu.addAction(self.open_img_file)


class ImgPanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.imageView = pg.ImageView()
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.imageView, 0, 0, 1, 9)
        self.setLayout(layout)
        self._current_data = None
        self.cmap = pg.ColorMap(np.linspace(0, 1, len(image_process.colorcube)), color=image_process.colorcube)
        self.imageView.setColorMap(self.cmap)

    def update_img(self, img, mask=None):
        if mask is not None:
            self._current_data = cv2.bitwise_and(img, img, mask=mask)
        else:
            self._current_data = img

        if len(img.shape) == 2:
            self.imageView.setImage(self._current_data.transpose(1, 0), autoRange=False)
        if len(img.shape) == 3:
            self.imageView.setImage(self._current_data.transpose(1, 0, 2), autoRange=False)

    def clear_img(self):
        self.imageView.clear()

    def get_img(self):
        return self._current_data


class PolarImagePanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        plot = pg.PlotItem()
        plot.setLabel(axis='left')
        plot.setLabel(axis='bottom')

        self.imageView = pg.ImageView(view=plot)
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.imageView)
        self.setLayout(self.layout)
        self.cmap = pg.ColorMap(np.linspace(0, 1, len(image_process.colorcube)), color=image_process.colorcube)
        self.imageView.setColorMap(self.cmap)

    def update_img(self, img):
        self._current_data = img
        if len(img.shape) == 2:
            self.imageView.setImage(self._current_data.transpose(1, 0), autoRange=False)
        if len(img.shape) == 3:
            self.imageView.setImage(self._current_data.transpose(1, 0, 2), autoRange=False)

    def clear_img(self):
        self.imageView.clear()

    def get_img(self):
        return self._current_data


class IntensityProfilePanel(QtWidgets.QWidget):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.plot_widget = ui_util.IntensityPlotWidget(title='Intensity Profile')
        self.plot_widget.setYScaling(False)
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.plot_widget)
        self.setLayout(self.layout)

        self.curr_dat = None
        self.prev_dat = None
        self.plot_widget.addLegend(offset=(-30, 30))
        self.plot_prev = self.plot_widget.plot(pen=pg.mkPen(100, 100, 100, width=2), name='previous')
        self.plot_curr = self.plot_widget.plot(pen=pg.mkPen(255, 0, 0, width=2), name='current')


    def update_graph(self, dat):
        if self.curr_dat is None:
            self.curr_dat = dat
            self.plot_curr.setData(self.curr_dat)
        else:
            self.prev_dat = self.curr_dat
            self.curr_dat = dat
            self.plot_curr.setData(self.curr_dat)
            self.plot_prev.setData(self.prev_dat)