import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks

paramK = np.loadtxt("./assets/Parameter_files/Kirkland_2010.txt")


def calculation(ds, q_start_num, q_end_num, element_nums, ratio, azavg, is_full_q, damping, rmax, dr, fit_at_q=None, N=None):
    element_nums = np.array(element_nums)
    element_nums = element_nums[element_nums != 0]
    ratio = np.array(ratio)
    ratio = ratio[ratio != 0]
    e_tot = np.sum(np.array(ratio))
    e_ratio = ratio / e_tot

    x = np.arange(q_start_num, q_end_num + 1)  # selected x ranges, end point = end point(eRDF) + 1
    Iq = azavg[q_start_num - 1:q_end_num]

    q = x * ds * 2 * np.pi

    s = q / 2 / np.pi
    s2 = s ** 2
    L = np.uint16(len(q))
    paramK_elems = paramK[element_nums, :]
    f = np.array([KirklandFactors(s2, paramK_elem) for paramK_elem in paramK_elems])
    fq = np.sum(f * e_ratio[:, None], axis=0)  # fq.shape = 2366,
    fq_sq = fq ** 2
    gq = np.sum(f ** 2 * e_ratio[:, None], axis=0)

    if is_full_q:
        AFrange = 0
    else:
        AFrange = int(2 / 3 * L)

    wi = np.ones((L, 1))
    wi[0:AFrange] = 0

    # added code
    if fit_at_q is not None:
        search_q = q[q < fit_at_q]
    else :
        search_q = q
    fit_at_q, qpos = search_q.max(), search_q.argmax()  # qmax = q_fix
    # end

    # qmax, qpos = q.max(), q.argmax()  # qmax = q_fix
    fqfit = gq[qpos]
    iqfit = Iq[qpos]

    if N is None:
        a1 = np.sum(wi * gq * Iq)
        a2 = np.sum(wi * Iq * fqfit)
        a3 = np.sum(wi * gq * iqfit)
        a4 = np.sum(wi) * fqfit * iqfit
        a5 = np.sum(wi * gq ** 2)
        a6 = 2 * np.sum(wi * gq * fqfit)
        a7 = np.sum(wi) * fqfit * fqfit
        N = (a1 - a2 - a3 + a4) / (a5 - a6 + a7)

    C = iqfit - N * fqfit

    Autofit = N * gq + C

    SS = np.sum((Iq - Autofit) ** 2);

    r = np.arange(0.01, rmax + dr, dr)  # rmax+dr to fit the eRDF

    phiq = ((Iq - Autofit) * s) / (N * fq_sq);
    phiq_damp = phiq * np.exp(-s2 * damping)

    Gr = 8 * np.pi * phiq_damp @ np.sin(q[:, None] * r) * ds

    return Iq, Autofit, phiq, phiq_damp, Gr, SS, fit_at_q, N


def KirklandFactors(s2, paramK_element):
    a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3 = paramK_element
    f = (a1 / (s2 + b1)) + (a2 / (s2 + b2)) + (a3 / (s2 + b3)) + (np.exp(-s2 * d1) * c1) + (np.exp(-s2 * d2) * c2) + (
                np.exp(-s2 * d3) * c3)
    return np.array(f)

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
