from abc import *
from epdfpy.file import load
import numpy as np
import logging
from epdfpy.calculate import image_process, pdf_calculator, elliptical_correction
import time
import os
import numpy as np
import hyperspy.api as hs
from itertools import product
import scipy.io
import cv2
from scipy.optimize import curve_fit
import pandas as pd

logger = logging.getLogger("Cube")


class Cube(metaclass=ABCMeta):
    def __init__(self, load_file_path: str = None, **kwargs):
        """
        Args:
            load_file_path:
            use_ready: True or False, for future use to reduce memory usage
            **kwargs:
        """


        self.load_file_path = load_file_path

        self.use_ready = kwargs.get("use_ready", False)

        self.kwargs = kwargs

        self.data = None
        self.blank = None  # MH edit 2023/07/04
        self.polar_data = None
        self.img_display = None
        self.center = [None, None]

        self.mask = None
        self.p_ellipse = None

        # if self.use_ready is False and load_file_path is not None:
        #     self.load_image(load_file_path)
        #     self.get_display_img()

    @abstractmethod
    def load_image(self, fp=None) -> np.array:
        # self.find_center()
        pass

    @abstractmethod
    def get_display_img(self):
        pass

    @abstractmethod
    def find_center(self):
        pass

    @abstractmethod
    def elliptical_fitting(self):
        pass

    @abstractmethod
    def elliptical_transformation(self):
        pass



class PDFCube(Cube):
    def __init__(self, load_file_path: str = None, filetype: str = None, **kwargs):
        """
        :param load_file_path:
        :param filetype: 'profile' or 'image' or None
        :param kwargs:
        """
        super().__init__(load_file_path, **kwargs)
        assert not np.logical_xor(load_file_path is None, filetype is None),\
            "Expected load_file_path and filetype simultaneously"

        self.azavg = None
        self.azvar = None
        self.preset_file_path = None

        self.pixel_start_n = None
        self.pixel_end_n = None
        self.ds = None
        self.data_quality = None
        self.data_quality_idx = None

        self.fit_at_q = None
        self.N = None
        self.damping = None
        self.rmax = None
        self.dr = None
        self.is_full_q = None

        self.r = None
        self.Gr = None

        self.all_q = None

        self.q = None
        self.full_q = None
        self.Iq = None

        self.SS = None
        self.phiq = None
        self.phiq_damp = None
        self.Autofit = None
        self.analyser = None
        self.element_nums = None
        self.element_ratio = None
        self.scattering_factor = None
        self.electron_voltage = None

        self.mask = None
        self.blank = None   #MH edit 2023/07/04
        self.blankpath = None
        self.origindata = None


        if filetype == 'profile':
            self.azavg = load._load_txt_img(self.load_file_path)
        elif filetype == 'image':
            self.data = load.load_diffraction_image(self.load_file_path)
            self.origindata = self.data.copy()
            self.get_display_img()


    def get_display_img(self):
        assert isinstance(self.data, np.ndarray) and self.data.ndim == 2, "Expected 2d numpy array"
        self.img_display = np.log(np.abs(self.data) + 1)
        self.img_display = self.img_display / self.img_display.max() * 255
        return self.img_display

    def load_image(self, fp=None) -> np.array:
        use_fp = fp or self.load_file_path
        if use_fp is None:
            logger.error("Need path")
            return
        self.data = load.load_diffraction_image(use_fp)
        assert self.data.ndim == 2, "Expected 2D array"
        return self.data

    def apply_blank(self):  #MH edit 2023/07/05
        print("Before min. value: {}".format(np.min(self.origindata)))
        if self.blank is None:
            print("No blank image")
        else:
            self.data = self.origindata.copy() - self.blank
            print(self.blankpath[self.blankpath.rfind('/')+1:] + ' noise subtracted')
        self.get_display_img()
        print("After min. value: {}".format(np.min(self.data)))
        return self.data

    def remove_blank(self): #MH edit 2023/07/05
        self.data = self.origindata.copy()
        print('Return to original')
        print(np.min(self.data))
        self.get_display_img()
        return self.data

    def find_center(self):
        self.center = list(image_process.calculate_center_gradient(self.img_display.copy(), self.center, self.mask))
        prev = [1,1]
        print('Checking center')
        cnt = 1
        while prev != self.center:
            prev = self.center.copy()
            self.center = list(image_process.calculate_center_gradient(self.img_display.copy(), self.center, self.mask))
            cnt += 1
            print('Failed. Readjusting...')
        else:
            print('Best fitted center', self.center)
        return self.center

    def calculate_azimuthal_average(self, version=0):
        assert self.data is not None, "You don't have img data"
        if version == 0:
            self.azavg = image_process.calculate_azimuthal_average(self.data, self.center, self.mask)
        elif version == 1:
            # py4dstem polar transformation
            self.azavg = elliptical_correction._fit_ellipse_amorphous_ring()
        elif version == 2:
            pass
        elif version == 3:
            pass
        return self.azavg

    def put_parameters(self, ds, pixel_start_n, pixel_end_n, element_nums, ratio, azavg, damping, rmax, dr, electron_voltage, fit_at_q=None, N=None, scattering_factor_type="Kirkland", fitting_range=None):
        # todo:
        self.ds = ds
        self.pixel_start_n = pixel_start_n
        self.pixel_end_n =  pixel_end_n
        self.element_nums = element_nums
        self.element_ratio = ratio
        self.azavg = azavg
        self.is_full_q = True

    def calculate_gr(self):
        assert self.ds is not None, "Put parameters first."
        pdf_calculator.calculation(
            self.ds, self.pixel_start_n, self.pixel_end_n, self.element_nums, self.ratio, self.azavg, self.is_full_q, self.damping, self.rmax, self.dr, self.electron_voltage, fit_at_q=None, N=None, scattering_factor_type="Kirkland", fitting_range=None)

    def elliptical_fitting(self):
        center, p_ellipse = elliptical_correction.elliptical_fitting_py4d_center_fixed(self.data, self.center[::-1], self.mask) # todo: center x,y
        self.p_ellipse = p_ellipse

    def elliptical_transformation(self, **kwargs):
        self.polar_data = elliptical_correction.polar_transformation_py4d(self.data, self.center, *self.p_ellipse, mask=self.mask, **kwargs)
        return self.polar_data



class FEMCube(Cube):
    def __init__(self, load_file_path: str = None, **kwargs):
        """
        Args:
            load_file_path:
            **kwargs:
            display_img_mod: "mean" or "var" or "median"
            repres_img_mod: "mean" or "median"
        """

        self.display_img_mod = kwargs.get("display_img_mod", None)
        self.repres_img_mod = kwargs.get("repres_img_mod", None)

        self.repres_img = None
        super(FEMCube, self).__init__(load_file_path, "image", **kwargs)

        self.CBEDstd = np.std(self.data, 0)
        self.ya, self.xa = np.meshgrid(np.arange(self.data.shape[1]), np.arange((self.data.shape[2])))

    def load_image(self, fp=None) -> np.array:
        use_fp = fp or self.load_file_path
        if use_fp is None:
            logger.error("Need path")
            return
        self.data = load.load_stem_image(use_fp)
        if self.data.ndim == 4:
            self.data = self.data.reshape((-1, self.data.shape[2], self.data.shape[3]))
        self.get_display_img()
        self.set_repres_img()
        return self.data

    def set_repres_img(self):
        if self.repres_img_mod == "mean":
            self.repres_img = np.mean(self.data, axis=0)
        elif self.repres_img_mod == "median":
            self.repres_img = np.median(self.data, axis=0)
        else:
            self.repres_img = np.median(self.data, axis=0)

    def intensity_refinement(self, mask=None):
        use_mask = self.choose_mask(mask, self.mask)
        ringInt = np.zeros(self.data.shape[0])
        for i in range(len(ringInt)):
            if use_mask is None:
                ringInt[i] = self.data[i].mean()
            else:
                ringInt[i] = self.data[i][use_mask.astype(bool)].mean()
        scaleInt = np.mean(ringInt) / ringInt
        for i in range(len(ringInt)):
            self.data[i] = self.data[i]*scaleInt[i]

    def elliptical_fitting_py4dstem(self, mask=None):
        use_mask = self.choose_mask(mask, self.mask)
        shp = np.array(self.repres_img.shape)
        center = shp / 2
        dist = np.hypot(*(shp - center)).astype(int)
        self.ellipse_rs = fem_calculation._fit_ellipse_amorphous_ring(self.repres_img, (self.repres_img.shape[0] / 2, self.repres_img.shape[1] / 2), (1, dist), mask=use_mask)
        A,B,C = self.ellipse_rs[0][2:5]
        self.A, self.B, self.C = 1, B/A, C/A

    def elliptical_fitting_matlab(self, mask=None):
        use_mask = self.choose_mask(mask, self.mask)


    def polar_transformation_matlab(self, mask=None, **kargs):
        elliptical_correction.polar_transformation_matlab(self.polarAll)


    def getVk(self):
        self.polarCBEDvar = np.mean((self.polarAll - self.polarRepresImg) ** 2, axis=0)
        self.radialMean = np.mean(self.polarRepresImg, axis=1)
        self.radialVar = np.mean(self.polarCBEDvar, axis=1)
        self.radialVarNorm = self.radialVar / (self.radialMean ** 2)
        return self.radialVarNorm

    def polar_transformation_py4dstem(self, mask=None, **kargs):
        use_mask = self.choose_mask(mask, self.mask)
        dr = kargs.get('dr', 2)
        dt = kargs.get('dt', None)
        if not dt:
            dt = kargs.get('dphi', None)
        if not dt:
            dt = 5
            dt = np.radians(dt)
        self.polarRepresImg = \
        fem_calculation._cartesian_to_polarelliptical_transform(self.repres_img, self.ellipse_rs[0][:5], mask=use_mask, dr=dr, dphi=dt)[0].T
        self.polarAll = np.zeros([self.data.shape[0], self.polarRepresImg.shape[0], self.polarRepresImg.shape[1]])
        self.polarAll = np.ma.array(data=self.polarAll)
        for i in range(self.data.shape[0]):
            self.polarAll[i] = \
            fem_calculation._cartesian_to_polarelliptical_transform(self.data[i], self.ellipse_rs[0][:5], mask=use_mask, dr=dr, dphi=dt)[0].T


    def get_display_img(self):
        assert isinstance(self.data, np.ndarray), "Expected numpy array"
        if self.display_img_mod == "mean":
            self.img_display = np.mean(self.data, axis=0)
        elif self.display_img_mod == "median":
            self.img_display = np.median(self.data, axis=0)
        elif self.display_img_mod == "var":
            self.img_display = np.var(self.data, axis=0)
        else:
            self.img_display = np.mean(self.data, axis=0)

    def find_center(self):
        pass

    def choose_mask(self, mask1, mask2):
        use_mask = None
        if mask1 is not None:
            use_mask = mask1
        elif mask2 is not None:
            use_mask = mask2
        return use_mask


if __name__ == "__main__":
    dc = FEMCube("C:\\Users\\vlftj\\Documents\FEM\\20220207\\220207_10_aSi_AD_220104_postB_area3_70x3_ss=5nm_C2=40um_alpha=0p63urad_spot11_2s_CL=245_bin=4_300kV.dm4")
    dc.intensity_refinement()
    dc.elliptical_fitting_py4dstem()
    dc.polar_transformation_py4dstem()
    print("Hello")

if __name__ == "__main__":
    dc = PDFCube("C:\\Users\\vlftj\\Documents\\sample41_Ta_AD\\Camera 230 mm Ceta 20201030 1709 0001_1_5s_1f_area01.mrc")
    dc.find_center()
    dc.calculate_azimuthal_average()
    print("Hello")