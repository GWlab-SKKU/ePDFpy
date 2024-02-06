import numpy as np
from epdfpy import definitions

paramK = np.loadtxt(definitions.KIRKLAND_PATH)
paramL = np.loadtxt(definitions.LOBATO_PATH)


def calculation(ds, pixel_start_n, pixel_end_n, element_nums, ratio, azavg, is_full_q, damping, rmax, dr, electron_voltage, fit_at_q=None, N=None, scattering_factor_type="Kirkland", fitting_range=None):
    assert len(element_nums) == len(ratio)
    if pixel_start_n is None or pixel_start_n==0:
        pixel_start_n = 1
    if pixel_end_n is None or pixel_end_n==0:
        pixel_end_n = len(azavg)

    element_nums = np.array(element_nums)
    for idx, element in enumerate(element_nums):
        if element == 0:
            ratio[idx] = 0
    element_nums = element_nums[element_nums != 0]
    ratio = np.array(ratio)
    ratio = ratio[ratio != 0]
    e_tot = np.sum(np.array(ratio))
    e_ratio = ratio / e_tot

    x = np.arange(pixel_start_n, pixel_end_n + 1)  # selected x ranges, end point = end point(eRDF) + 1
    # Iq = azavg[px_start_num-1:px_end_num]        # Indexing number
    Iq = azavg[pixel_start_n: pixel_end_n + 1]

    q = x * ds * 2 * np.pi

    s = q / 2 / np.pi
    s2 = s ** 2

    if scattering_factor_type == "Kirkland":
        paramK_elems = paramK[element_nums, :]
        f = np.array([KirklandFactors(s2, paramK_elem) for paramK_elem in paramK_elems])
    elif scattering_factor_type == "Lobato":
        paramL_elems = paramL[element_nums, :]
        f = np.array([LobatoFactors(s2, paramL_elem) for paramL_elem in paramL_elems])
    f = f * calculate_relativistic(electron_voltage)

    fq = np.sum(f * e_ratio[:, None], axis=0)
    fq_sq = fq ** 2
    gq = np.sum(f ** 2 * e_ratio[:, None], axis=0)

    L = np.uint16(len(q))
    # if is_full_q:
    #     AFrange = 0
    # else:
    #     AFrange = int(2 / 3 * L)

    wi = np.ones((L))
    if fitting_range is not None:
        # wi = np.ones((L, 1))
        wi = np.zeros((L))
        # wi[0:AFrange] = 0
        q_to_x_factor = 1 / (ds * 2 * np.pi)
        l = np.int(np.round(q_to_x_factor * fitting_range[0]))
        r = np.int(np.round(q_to_x_factor * fitting_range[1]))
        wi[l:r] = 1

    # added code
    if fit_at_q is not None:
        search_q = q[q <= fit_at_q + (ds/2)*(2*np.pi)]
    else:
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

    return q, r, Iq, Autofit, phiq, phiq_damp, Gr, SS, fit_at_q, N


def calculate_relativistic(voltage):
    if voltage == '':
        voltage = 0.0
    voltage = float(voltage)
    c = 2.998e8
    relvelocity = c * (1 - 1 / (1 + voltage/511)**2 ) ** 0.5
    mass_e_relative = 1 / (1-(relvelocity**2/c**2)) ** 0.5
    return mass_e_relative

def rescaling_Iq(q_start_num, q_end_num, azavg, ds):
    x = np.arange(q_start_num, q_end_num + 1)  # selected x ranges, end point = end point(eRDF) + 1
    Iq = azavg[q_start_num - 1:q_end_num]

    q = x * ds * 2 * np.pi
    return q, Iq

def pixel_to_q(args, ds):
    return np.array(args) * ds * 2 * np.pi

def q_to_pixel(args, ds):
    return np.round(np.array(args) / ds / 2 / np.pi).astype(int)

def _calculation_with_q(ds, q, Iq, element_nums, ratio, is_full_q, damping, rmax, dr, fit_at_q=None, N=None):
    element_nums = np.array(element_nums)
    for idx, element in enumerate(element_nums):
        if element == 0:
            ratio[idx] = 0
    element_nums = element_nums[element_nums != 0]
    ratio = np.array(ratio)
    ratio = ratio[ratio != 0]
    e_tot = np.sum(np.array(ratio))
    e_ratio = ratio / e_tot

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
        search_q = q[q <= fit_at_q + ds/2]
    else:
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

    # Gr = 8 * np.pi * phiq_damp @ np.sin(q[:, None] * r) * ds
    Gr = 8 * np.pi * phiq_damp @ np.sin(s[:, None] * r) * ds

    return q, r, Iq, Autofit, phiq, phiq_damp, Gr, SS, fit_at_q, N

def LobatoFactors(s2, paramL_element):
    A1, A2, A3, A4, A5, B1, B2, B3, B4, B5 = paramL_element
    f = (A1 * (s2 * B1 + 2))/((s2 * B1 + 1)**2) + (A2 * (s2 * B2 + 2))/((s2 * B2 + 1)**2) + \
         (A3 * (s2 * B3 + 2))/((s2 * B3 + 1)**2) + (A4 * (s2 * B4 + 2))/((s2 * B4 + 1)**2) + \
         (A5 * (s2 * B5 + 2))/((s2 * B5 + 1)**2)
    return np.array(f)

def KirklandFactors(s2, paramK_element):
    a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3 = paramK_element
    f = (a1 / (s2 + b1)) + (a2 / (s2 + b2)) + (a3 / (s2 + b3)) + (np.exp(-s2 * d1) * c1) + (np.exp(-s2 * d2) * c2) + (
                np.exp(-s2 * d3) * c3)
    # f1 = ((s2+b1_1).\a1_1)+((s2+b2_1).\a2_1)+((s2+b3_1).\a3_1)+(exp(-s2.*d1_1).*c1_1)+(exp(-s2.*d2_1).*c2_1)+(exp(-s2.*d3_1).*c3_1);
    return np.array(f)