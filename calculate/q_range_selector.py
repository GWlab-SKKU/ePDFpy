from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks
import numpy as np


def find_first_peak(azavg, derivative=0):
    # flattening
    first_peak_idx = 1
    for i in range(len(azavg)):
        if azavg[i] != 0:
            first_peak_idx = i
            break

    range_slice = slice(first_peak_idx, first_peak_idx + int(len(azavg) * 0.3))
    filtered_part = gaussian_filter1d(azavg[first_peak_idx:first_peak_idx + int(len(azavg) * 0.3)], sigma=2)
    azavg2 = azavg.copy()
    azavg2[range_slice] = filtered_part

    # first order
    x = azavg2[0:int(len(azavg) * 0.3)]
    low_peaks, _ = find_peaks(-x, distance=20)

    if len(low_peaks) > 0 and (derivative is not 2):
        return low_peaks[0]
    if derivative == 1:
        return None

    # second order
    else:
        x = np.gradient(azavg2, 0.1)
        # peaks, _ = find_peaks(x, distance=20)
        low_peaks, _ = find_peaks(-x, distance=20)

        # peaks = peaks[azavg2[peaks] != 0]
        low_peaks = low_peaks[azavg2[low_peaks] != 0]

        if len(low_peaks) > 0:
            return low_peaks[0]
        else:
            return None


def find_multiple_peaks(azav):
    azavg = azav.copy()
    azavg = azavg[:int(len(azavg)/3)]

    gaussian_sigma = 3

    # find first non-zero idx
    first_nonzero_idx = find_first_nonzero_idx(azavg)

    # smoothing
    azavg[first_nonzero_idx:] = gaussian_filter1d(azavg[first_nonzero_idx:],gaussian_sigma)

    # find first derivative peaks
    f_low_peaks, _ = find_peaks(-azavg, distance=10)
    f_high_peaks, _ = find_peaks(azavg, distance=10)
    f_mixed_peaks = np.concatenate([f_low_peaks,f_high_peaks])
    # plt.plot(azavg)
    # plt.scatter(f_mixed_peaks,azavg[f_mixed_peaks])

    # calculate derivative
    first_derivative = np.zeros(len(azavg))
    first_derivative[first_nonzero_idx:] = np.gradient(azavg[first_nonzero_idx:])
    # plt.plot(first_derivative)

    # smoothing
    # first_derivative[first_nonzero_idx:] = gaussian_filter1d(first_derivative[first_nonzero_idx:],gaussian_sigma)

    # find second derivative peaks
    s_low_peaks, _ = find_peaks(-first_derivative, distance=10)
    s_high_peaks, _ = find_peaks(first_derivative, distance=10)
    s_mixed_peaks = np.concatenate([s_low_peaks,s_high_peaks])
    return f_mixed_peaks,s_mixed_peaks

def find_first_nonzero_idx(azavg):
    for i in range(len(azavg)):
        if azavg[i] != 0:
            return i
