# 1st edit 210817
# 2nd edit 210819 after LM
# 3rd edit 210907 solving q_s
# 4th edit 210910 applying to others
# 5th edit 210913 nk, ns plots
# 6th edit 210914 w/o n plot. Just RDF cal
# 7th edit 211001 added qk 17.5~22.5 + judge1, judge2
# 8th edit 211004 filename: sample#_area# modified + ootest prodece
# 9th edit 211006 noise area change
# 10th edit 211023 r 1~1.7 oscillation minimum
# 11st edit 211028 1 area checking
# 12nd edit 211108 damping -> q2
# 13th edit 211110 optimized
# 14th edit 220215 Major update: q_min-> pixels + Gr result checked + results saved in files
# 15th edit 220216 Change to txt load + json (or png) min pixel load
# 16th edit 220218 Change after LM -> data_quality: all, save in TEM diffraction folder
# 17th edit 220219 Fixing the index problems
# 18th edit 220221 Change from loop to matrix + fix nan prolem
# 19th edit 220222 Judge3 == 3 + Found out G(r) is saved wrong... fixed. + ePDF origin included
# 20th edit 220223 Added ePDF calculation + indexing probelm + separate azavg / newind file module.
# 21st edit 220224 Delete unnecessary variable to get more free memory
# 22nd edit 220226 Edited for ePDFpy application
# 23rd edit 220307 Reduced time -> np.dot to matmul + no vstack
# 24th edit 220308 Reduced time -> judge_oo from matrix calculation
# 25th edit 230119 Added relativistic effect

import numpy as np
import time
import pandas as pd
import os
from epdfpy import definitions
from epdfpy.calculate.pdf_calculator import calculate_relativistic  # 25th edit

kirkland_fp = definitions.KIRKLAND_PATH
lobato_fp = definitions.LOBATO_PATH


def Kirkland_File(file_name):
    with open(file_name, 'r') as data:
        a1 = []
        b1 = []
        a2 = []
        b2 = []
        a3 = []
        b3 = []
        c1 = []
        d1 = []
        c2 = []
        d2 = []
        c3 = []
        d3 = []
        for line in data:
            p = line.split()
            a1.append(float(p[0]))
            b1.append(float(p[1]))
            a2.append(float(p[2]))
            b2.append(float(p[3]))
            a3.append(float(p[4]))
            b3.append(float(p[5]))
            c1.append(float(p[6]))
            d1.append(float(p[7]))
            c2.append(float(p[8]))
            d2.append(float(p[9]))
            c3.append(float(p[10]))
            d3.append(float(p[11]))

    return a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3


a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3 = Kirkland_File(kirkland_fp)


def Lobato_file(file_name):
    with open(file_name, 'r') as data:
        a1 = []
        a2 = []
        a3 = []
        a4 = []
        a5 = []
        b1 = []
        b2 = []
        b3 = []
        b4 = []
        b5 = []
        for line in data:
            p = line.split()
            a1.append(float(p[0]))
            a2.append(float(p[1]))
            a3.append(float(p[2]))
            a4.append(float(p[3]))
            a5.append(float(p[4]))
            b1.append(float(p[5]))
            b2.append(float(p[6]))
            b3.append(float(p[7]))
            b4.append(float(p[8]))
            b5.append(float(p[9]))

    return a1, a2, a3, a4, a5, b1, b2, b3, b4, b5


A1, A2, A3, A4, A5, B1, B2, B3, B4, B5 = Lobato_file(lobato_fp)


def Kirkland_factor(cf, pix_min, pix_max):
    pix = np.arange(pix_min, pix_max + 1, 1)
    s = cf * pix
    s2 = s ** 2

    Kf = np.zeros((len(a1), len(pix)))

    for j in np.arange(0, len(a1), 1):
        element = j
        for i in np.arange(1, len(pix), 1):
            Kf[j, i] = (a1[element]) / (b1[element] + s2[i]) + (c1[element] * np.exp((-1) * d1[element] * s2[i])) + (
                a2[element]) / (b2[element] + s2[i]) + (c2[element] * np.exp((-1) * d2[element] * s2[i])) + (
                           a3[element]) / (b3[element] + s2[i]) + (c3[element] * np.exp((-1) * d3[element] * s2[i]))

    return Kf


def Lobato_factor(cali, pix_min, pix_max):
    pix = np.arange(pix_min, pix_max + 1, 1)
    q = cali * pix
    q2 = q ** 2

    Lf = np.zeros((len(A1), len(pix)))

    for j in np.arange(0, len(a1), 1):
        element = j
        for i in np.arange(0, len(pix), 1):
            Lf[j, i] = (A1[element] * ((2 + (B1[element] * q2[i])) / (1 + (B1[element] * q2[i])) ** 2)) + (
                    A2[element] * ((2 + (B2[element] * q2[i])) / (1 + (B2[element] * q2[i])) ** 2)) + (
                               A3[element] * ((2 + (B3[element] * q2[i])) / (1 + (B3[element] * q2[i])) ** 2)) + (
                               A4[element] * ((2 + (B4[element] * q2[i])) / (1 + (B4[element] * q2[i])) ** 2)) + (
                               A5[element] * ((2 + (B5[element] * q2[i])) / (1 + (B5[element] * q2[i])) ** 2))

    return Lf


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.mkdir(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def Autofit(Iq, qkran_start, qkran_end, qkran_step, pixran_start, pixran_end, pixran_step, Elem, Rat, pixel_start_n,
            Calibration_factor, Damping_factor, Noise_threshold, Select, use_lobato, Voltage):# 25th edit
    #######################preset######################
    if Noise_threshold == '':
        Noise_threshold = 1.0
    print('Autofit started')
    toc = time.time()
    cal = float(Calibration_factor)
    cali = 2 * np.pi * cal

    qk_list = np.arange(qkran_end, qkran_start - qkran_step, -qkran_step)  # Scanning range of qk
    pixran = np.arange(pixran_end, pixran_start - pixran_step, -pixran_step)  # Scanning range of Max. pixel

    if Select == '':
        Select = int(10)

    dn = 5  # N pm range
    Noise_level = float(Noise_threshold)  # Standard for Max. value of G(r) in r<1A
    damp = float(Damping_factor)  # Usually .15
    #############Pixel define###################3
    pix_max = 2400
    pix = np.arange(0, pix_max + 1, 1)  # Just in case, calculate large enough range
    Q = cal * 2 * np.pi * pix
    q = cal * pix
    q2 = q ** 2
    rmax = 10
    dr = 0.01
    r = np.arange(dr, rmax + dr, dr)
    dq = q[1] - q[0]

    if use_lobato:
        Kf = Lobato_factor(cal, 0, pix_max)  # Lobato scattering factor
        print('Lobato')
    else:
        Kf = Kirkland_factor(cal, 0, pix_max)  # Kirkland scattering factor
        print('Kirkland')

    Kf = Kf * calculate_relativistic(Voltage)  # 25th edit
    Ratio = Rat / np.sum(Rat)

    kf1, kf2, kf3, kf4, kf5 = Kf[Elem[0], :], Kf[Elem[1], :], Kf[Elem[2], :], Kf[Elem[3], :], Kf[Elem[4],
                                                                                              :]  # Defining scattering factor for each elements
    kf_arr = np.vstack([kf1, kf2, kf3, kf4, kf5])
    kf_Q = np.dot(kf_arr.T, Ratio.T)
    kf2_Q = np.dot(kf_arr.T ** 2, Ratio.T)  # <f^2>
    kf_2_Q = kf_Q ** 2  # <f>^2

    ##################Autofit start###########################
    Ncan = np.arange(0, 1000, 1) + 1  # N fitting range
    intensity = Iq
    pix_m = pixel_start_n
    q_min = pix_m * cali

    seed = np.ones(
        (len(qk_list) * len(pixran), np.max(pixran) + 1 - pix_m))  # For matching up the size of stacking matrix
    seed = seed * np.nan
    data_Q, data_q, intensity_Q, kf2, kf_2 = seed.copy(), seed.copy(), seed.copy(), seed.copy(), seed.copy()
    Nklist = np.array([0])
    params = np.ones((len(qk_list) * len(pixran), 3))
    line = 0
    for max_pix in pixran:
        #        print(max_pix)
        for q_max in qk_list:
            if q_max > max_pix * cali:
                continue

            ap = np.arange(int(pix_m), int(max_pix + 1),
                           1)  ##Edited 2/21 -> Applied for 'newind' azav files. Slicing Min ~ Max pixels.

            temp_Q, temp_q, temp_i, temp_k, temp_k2 = Q[ap], q[ap], intensity[ap], kf2_Q[ap], kf_2_Q[ap]

            Q_max = int(np.round(q_max / cali))
            Q_min = int(np.round(q_min / cali))
            if (Q_max - int(q_min / cali)) >= len(temp_i[~np.isnan(temp_i)]):
                continue
            iq = temp_i[Q_max - Q_min]  # Iqfit
            kk = temp_k[Q_max - Q_min]  # ffit

            temp_i = temp_i - iq
            temp_k = temp_k - kk

            ###########################################################################
            fit_r = np.arange(int((2 * np.pi) / cali) - Q_min, Q_max - Q_min + 1,
                              1)  # tail fitting from 2pi

            ############################### N fitting from 2pi for all possible cases ##########################
            Nseed = np.ones((1000, len(fit_r)))
            nIq = Nseed * (temp_i[fit_r])
            nkk = (Nseed.T * Ncan).T * (temp_k[fit_r])
            costk = np.sum((nIq - nkk) ** 2, axis=1)
            AutoNk = costk.argmin() + 1
            Nklist = np.append(Nklist, AutoNk)

            ######################################################################################################
            ################### The actual data used for calculation###################
            data_Q[line, :len(ap)] = temp_Q
            data_q[line, :len(ap)] = temp_q
            kf2[line, :len(ap)] = temp_k
            intensity_Q[line, :len(ap)] = temp_i
            kf_2[line, :len(ap)] = temp_k2
            params[line] = [pix_m, max_pix, np.round(q_max, 2)]
            del (temp_Q, temp_q, temp_i, temp_k, temp_k2, nIq, nkk, costk)
            line = line + 1

            ######################## Adding another axis to data matrix (N axis) ###########################################
    listup = np.ones((2 * dn + 1, line, len(data_Q[0])))
    paramT = np.ones((2 * dn + 1, line, len(params[0])))

    data_Q = data_Q[:line] * listup
    data_q = data_q[:line] * listup
    intensity_Q = intensity_Q[:line] * listup
    kf2 = kf2[:line] * listup
    kf_2 = kf_2[:line] * listup
    Nklist = np.delete(Nklist, 0)
    Nk, Nbase = listup.copy(), listup[0]
    params = params[:line] * paramT
    del (listup, paramT)
    for i in range(0, 2 * dn + 1, 1):
        Nk_add = Nbase.T
        Nk_add[:, ] = Nklist + (i - dn)
        Nk[i] = Nk_add.T

    Params = np.dstack((params, np.transpose(Nk, (2, 1, 0))[0].T))  # Parameters: Min.pix, Max.pix, qk, N
    Kf2, Kf_2 = kf2 * Nk, kf_2 * Nk  # N<f^2>, N<f>^2
    damping = np.exp(-1 * damp * (data_q ** 2))

    ###################### Reduced intensity function #############################
    Phik = ((intensity_Q - Kf2) / Kf_2) * data_q
    Phik_d = Phik * damping
    del (damping, intensity_Q, Nklist, Nbase, seed, params, kf2, kf_2)
    ####################### Fourier transform ###########################
    R = r[np.newaxis, :]
    qtr = data_Q[0][:, :, np.newaxis]
    QR = np.dot(np.nan_to_num(qtr[0]), R)
    del (R, qtr)
    Sin_QR = np.sin(QR)
    del (QR)
    Gk = 8 * np.pi * np.matmul(np.nan_to_num(Phik_d), Sin_QR) * dq
    del (data_q, Sin_QR)
    ###########################################################
    ####################### Filtering ########################################

    ################# Reshaping all result to 2D matrix #######################
    Params = Params.reshape(int(Params.size / len(Params[0][0])), len(Params[0][0]))
    Gk = Gk.reshape(int(Gk.size / len(r)), len(r))
    Phi_d = Phik_d.reshape(int(Phik_d.size / len(data_Q[0][0])), len(data_Q[0][0]))
    Phi = Phik.reshape(int(Phik.size / len(data_Q[0][0])), len(data_Q[0][0]))
    data_Q = data_Q.reshape(int(data_Q.size / len(data_Q[0][0])), len(data_Q[0][0]))
    total_n = len(Gk)

    ################### Defining selection condition #########################
    noise_area = np.arange(0, 100, 1)
    fir_peak_area = np.arange(100, 250, 1)
    oo_qarea = np.arange(200, 350, 1)
    noise_r = r[:np.max(noise_area)+1]
    # judge_noise = np.max(np.abs(Gk[:, noise_area]), axis=1)  # S.C 1
    judge_noise = np.max((Gk[:, noise_area]), axis=1)  # S.C 1 #MH edit 2023/07/14
    judge_1st = np.max(Gk[:, fir_peak_area], axis=1)  # S.C 2
    # judge_std = np.std(Gk[:, noise_area], axis=1)  # Grading = std
    judge_std = np.linalg.lstsq(noise_r[:,np.newaxis],np.array(Gk[:,noise_area]).T)[1]
    # asym = np.polynomial.polynomial.polyfit(noise_area,Gk[:, noise_area].T,1)
    #
    # judge_asym = np.std(Gk[:,noise_area]-np.poly1d(asym)(noise_area),axis=1)
    # print(judge_asym)

    ############ number of oo_peak ################
    gra = np.gradient(Gk[:, oo_qarea])[1]
    gracheck = np.sign(gra[:, :-1] * gra[:, 1:])
    grad_mask = gracheck == -1
    judge_oo = np.sum((Gk[:, 200:349] * grad_mask) < 0, axis=1)

    Autofit = []

    for i in range(len(Params)):
        Auto_temp = []
        Auto_temp.append(Params[i][0])
        Auto_temp.append(Params[i][1])
        Auto_temp.append(Params[i][2])
        Auto_temp.append(Params[i][3])
        Auto_temp.append(np.array(data_Q[i][~np.isnan(data_Q[i])]))
        Auto_temp.append(np.array(Phi[i][~np.isnan(Phi[i])]))
        Auto_temp.append(np.array(Phi_d[i][~np.isnan(Phi_d[i])]))
        Auto_temp.append(np.array(r))
        Auto_temp.append(np.array(Gk[i]))
        Auto_temp.append(judge_noise[i])
        Auto_temp.append(judge_1st[i])
        Auto_temp.append(judge_oo[i])
        Auto_temp.append(judge_std[i])
        Autofit.append(Auto_temp)
    del (Params, data_Q, Phi, Phi_d, Gk, judge_noise, judge_1st, judge_oo, judge_std)

    Results = pd.DataFrame(Autofit,
                           columns=['Min pix', 'Max pix', 'qk', 'N', 'Q', 'Phi(q)', 'Phi_d(q)', 'r', 'G(r)', 'judge1',
                                    'judge2', 'judge3', 'judge4'])
    Pre_Results = Results.sort_values('judge4')  # Ordering by std
    # Results = Pre_Results[Pre_Results['judge1'] < Noise_level]  # S.C 1
    # print('original')
    Results = Pre_Results.nsmallest(int((Noise_level / 100) * len(Pre_Results)),'judge1',keep='all') # S.C 1 # MH edit 2023/07/13
    Results = Results.sort_values('judge4')
    Results = Results[Results['judge1'] < Results['judge2']]  # S.C 2
    Results = Results[Results['judge3'] == 3]  # S.C 3
    qualitycheck = len(Results)  # If good sample, this # is large

    if qualitycheck != 0:  # For good samples which passed S.C 1
        if qualitycheck < Select:
            Candidates = Results[:qualitycheck].sort_values(['Max pix', 'qk', 'N'],
                                                            ascending=[False, False, False]).to_numpy()
        if qualitycheck >= Select:
            Candidates = Results[:Select].sort_values(['Max pix', 'qk', 'N'],
                                                      ascending=[False, False, False]).to_numpy()

    else:  # For bad samples which didn't pass the S.C 1
        Pre_Results2 = Pre_Results[Pre_Results['judge1'] < Pre_Results['judge2']]
        qualitycheck2 = len(Pre_Results2)
        if qualitycheck2 < Select:
            Candidates = Pre_Results2[:qualitycheck2].sort_values(['Max pix', 'qk', 'N'],
                                                                  ascending=[False, False, False]).to_numpy()
        if qualitycheck2 >= Select:
            Candidates = Pre_Results2[:Select].sort_values(['Max pix', 'qk', 'N'],
                                                           ascending=[False, False, False]).to_numpy()
        if len(Pre_Results2) == 0:
            Candidates = Pre_Results[:Select].sort_values(['Max pix', 'qk', 'N'],
                                                          ascending=[False, False, False]).to_numpy()
        del (Results, Pre_Results, Pre_Results2)
    tic = time.time()
    elapsed = str(tic - toc) + ' s'
    print('Autofit finished')
    print('Elapsed time:', elapsed)

    return Candidates, qualitycheck, total_n  # Candidate: top 10 results in numpy array / qualitycheck: How many that passed the filters