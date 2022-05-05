from abc import *
from file import load
import numpy as np
import logging
from calculate import image_process, pdf_calculator, fem_calculation
import time
import os
import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt
from itertools import product
import scipy.io
import cv2
from scipy.optimize import curve_fit
import pandas as pd

logger = logging.getLogger("Cube")


class Cube(metaclass=ABCMeta):
    def __init__(self, load_file_path: str = None, filetype: str = None, **kwargs):
        """

        Args:
            load_file_path:
            filetype: "image", "profile", "preset"
            use_ready:
            **kwargs:
        """
        assert not np.logical_xor(load_file_path is None, filetype is None),\
            "Expected load_file_path and filetype simultaneously"

        self.load_file_path = load_file_path
        self.filetype = filetype

        self.use_ready = kwargs.get("use_ready", False)

        self.kwargs = kwargs

        self.data = None
        self.img_display = None
        self.center = [None, None]

        self.mask = None

        if self.use_ready is False and load_file_path is not None:
            self.load_image(load_file_path)
            self.get_display_img()

    @abstractmethod
    def load_image(self, fp=None) -> np.array:
        pass

    @abstractmethod
    def get_display_img(self):
        pass

    @abstractmethod
    def find_center(self):
        pass


class PDFCube(Cube):
    def __init__(self, load_file_path: str = None, filetype: str = None, **kwargs):
        self.azavg = None
        self.azvar = None
        self.img_raw = None

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

        self.original_data = None
        self.mask = None

        super().__init__(load_file_path, filetype, **kwargs)

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
        self.img_raw = self.data.copy()
        return self.img_raw

    def find_center(self):
        self.center = list(image_process.calculate_center_gradient(self.img_raw.copy(), self.mask))
        return self.center

    def calculate_azimuthal_average(self, version=0):
        assert self.img_raw is not None, "You don't have img data"
        if version == 0:
            self.azavg = image_process.calculate_azimuthal_average(self.img_raw, self.center, self.mask)
        elif version == 1:
            # py4dstem polar transformation
            self.azavg = fem_calculation.fit_ellipse_amorphous_ring()
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
        self.ellipse_rs = fem_calculation.fit_ellipse_amorphous_ring(self.repres_img, (self.repres_img.shape[0]/2,self.repres_img.shape[1]/2), (1,dist), mask=use_mask)
        A,B,C = self.ellipse_rs[0][2:5]
        self.A, self.B, self.C = 1, B/A, C/A

    def elliptical_fitting_matlab(self, mask=None):
        use_mask = self.choose_mask(mask, self.mask)
        def fit_func(x, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11):
            Rsq = (x[:, 0] - c1) ** 2 + c4 * (x[:, 0] - c1) * (x[:, 1] - c2) + c3 * (x[:, 1] - c2) ** 2
            rs = c5 + \
                 c6 * np.exp(-1 / 2 / c7 ** 2 * Rsq) + \
                 c9 * np.exp(-1 / 2 / c10 ** 2 * np.abs(c8 - np.sqrt(Rsq)) ** 2) * (c8 ** 2 > Rsq) + \
                 c9 * np.exp(-1 / 2 / c11 ** 2 * np.abs(c8 - np.sqrt(Rsq)) ** 2) * (c8 ** 2 < Rsq)
            return rs

        if use_mask is None:
            use_mask = np.ones(self.data.shape)
        skipFit = [11, 1]
        stackSize = self.data.shape
        inds2D = np.ravel_multi_index([self.ya, self.xa], stackSize[1:3])
        basis = np.array((self.ya.reshape(-1), self.xa.reshape(-1))).transpose()
        p0 = 1.0e+03 * np.array([0.2565, 0.2624, 0.0010, 0.0001, 0.0, 1.4273, 0.0535,
                                 0.1198, 0.2895, 0.0118, 0.0104])
        lb = [0, 0, 0.5, -0.5,
              0, 0, 0, 0, 0, 0, 0];
        ub = [stackSize[1], stackSize[2], 2, 0.5
            , np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf];
        coefsInit = p0
        for a0 in range(0, len(skipFit)):
            maskFit = (inds2D % skipFit[a0]) == 0;
            coefsInit = curve_fit(fit_func,
                                  basis[np.logical_and(use_mask.reshape(-1), maskFit.reshape(-1))],
                                  self.repres_img[np.logical_and(use_mask, maskFit)],
                                  p0=coefsInit,
                                  bounds=(lb, ub)
                                  )[0]
        self.A = 1.0
        self.B = coefsInit[3]
        self.C = coefsInit[2]
        coefs = [coefsInit[1], coefsInit[0], self.A, self.C, self.B]

        self.ellipse_rs = [coefs, coefsInit]

    def polar_transformation_matlab(self, mask=None, **kargs):
        fem_calculation.polar_transformation_matlab(self.polarAll)
        # pixelSize = kargs.get('pixelSize', 1)
        # rSigma = kargs.get('rSigma', 0.1)
        # tSigma = kargs.get('tSigma', 1)
        # rMax = kargs.get('rMax', 240)
        # dr = kargs.get('dr', 2)
        # dt = kargs.get('dt', 5)
        #
        # use_mask = self.choose_mask(mask, self.mask)
        #
        # polarRadius = np.arange(0, rMax, dr) * pixelSize;
        # polarTheta = np.arange(0, 360, dt) * (np.pi / 180)
        # ya, xa = np.meshgrid(np.arange(self.data.shape[1]), np.arange(self.data.shape[2]))
        #
        # coefs = [self.ellipse_rs[0][0], self.ellipse_rs[0][1], self.C, self.B]
        # xa = self.xa - coefs[0]
        # ya = self.ya - coefs[1]
        # # Correction factors
        # if abs(self.C) > -6:
        #     p0 = -np.arctan((1 - self.B + np.sqrt((self.B - 1) ** 2 + self.C ** 2)) / self.C);
        # else:
        #     p0 = 0;
        #
        # a0 = np.sqrt(2 * (1 + self.C + np.sqrt((self.C - 1) ** 2 + self.B ** 2)) / (4 * self.C - self.B ** 2))
        # b0 = np.sqrt(2 * (1 + self.C - np.sqrt((self.C - 1) ** 2 + self.B ** 2)) / (4 * self.C - self.B ** 2))
        # ratio = b0 / a0;
        # m = [[ratio * np.cos(p0) ** 2 + np.sin(p0) ** 2,
        #       -np.cos(p0) * np.sin(p0) + ratio * np.cos(p0) * np.sin(p0)],
        #      [-np.cos(p0) * np.sin(p0) + ratio * np.cos(p0) * np.sin(p0),
        #       np.cos(p0) ** 2 + ratio * np.sin(p0) ** 2]]
        # m = np.array(m)
        # ta = np.arctan2(m[1, 0] * xa + m[1, 1] * ya, m[0, 0] * xa + m[0, 1] * ya)
        # ra = np.sqrt(xa ** 2 + coefs[2] * ya ** 2 + coefs[3] * xa * ya) * b0
        #
        # # Resamping coordinates
        # Nout = [len(polarRadius), len(polarTheta)];
        # rInd = (np.round((ra - polarRadius[0]) / dr)).astype(np.uint)
        # tInd = (np.mod(np.round((ta - polarTheta[0]) / (dt * np.pi / 180)), Nout[1])).astype(np.uint)
        # sub = np.logical_and(rInd <= Nout[0] - 1, rInd >= 0)
        # rtIndsSub = np.array([rInd[sub], tInd[sub]]).T;
        #
        # polarNorm = fem_calculation.accum(rtIndsSub, np.ones(np.sum(sub)))
        # self.polarRepresImg = fem_calculation.accum(rtIndsSub, self.repres_img[sub])
        # self.polarRepresImg = self.polarRepresImg / polarNorm
        # if use_mask is None:
        #     use_mask = np.ones(self.data.shape[1:])
        # polarMask = fem_calculation.accum(rtIndsSub, use_mask[sub]) == 0
        #
        # self.polarRepresImg = np.ma.masked_array(self.polarRepresImg, polarMask)
        # self.polarAll = np.zeros((self.data.shape[0], Nout[0], Nout[1]))
        # for a0 in range(self.data.shape[0]):
        #     CBED = self.data[a0];
        #     polarCBED = fem_calculation.accum(rtIndsSub, CBED[sub])
        #     polarCBED = polarCBED / polarNorm
        #     self.polarAll[a0] = polarCBED;

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
        fem_calculation.cartesian_to_polarelliptical_transform(self.repres_img, self.ellipse_rs[0][:5], mask=use_mask, dr=dr, dphi=dt)[0].T
        self.polarAll = np.zeros([self.data.shape[0], self.polarRepresImg.shape[0], self.polarRepresImg.shape[1]])
        self.polarAll = np.ma.array(data=self.polarAll)
        for i in range(self.data.shape[0]):
            self.polarAll[i] = \
            fem_calculation.cartesian_to_polarelliptical_transform(self.data[i], self.ellipse_rs[0][:5], mask=use_mask, dr=dr, dphi=dt)[0].T


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