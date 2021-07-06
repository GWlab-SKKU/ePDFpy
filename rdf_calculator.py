import numpy as np

paramK = np.loadtxt("./assets/Parameter_files/Kirkland_2010.txt")


def calculation(ds, q_start_num, q_end_num, element_nums, ratio, azavg, is_full_q, damping, rmax, dr):
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
    qmax, qpos = q.max(), q.argmax()  # qmax = q_fix
    fqfit = gq[qpos]
    iqfit = Iq[qpos]

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

    return Iq, Autofit, phiq, phiq_damp, Gr, SS


def KirklandFactors(s2, paramK_element):
    a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3 = paramK_element
    f = (a1 / (s2 + b1)) + (a2 / (s2 + b2)) + (a3 / (s2 + b3)) + (np.exp(-s2 * d1) * c1) + (np.exp(-s2 * d2) * c2) + (
                np.exp(-s2 * d3) * c3)
    return np.array(f)
