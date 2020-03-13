from qutip import *
import numpy as np
from scipy.special import kv
import Single_small_junction as ssj
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def calcSpectrum(level_num, N, E_l, E_c, E_j, I0, I_period, I_array):
    phi_ext = (I_array - I0) / I_period * 2 * np.pi
    if level_num < 3:
        level_num = 3
    phi_num = phi_ext.size

    n0x = np.zeros((phi_num, level_num - 1))
    n1x = np.zeros((phi_num, level_num - 2))
    n2x = np.zeros((phi_num, level_num - 3))
    phi0x = np.zeros((phi_num, level_num - 1))
    phi1x = np.zeros((phi_num, level_num - 2))
    phi2x = np.zeros((phi_num, level_num - 3))
    freq0x = np.zeros((phi_num, level_num - 1))
    freq1x = np.zeros((phi_num, level_num - 2))
    freq2x = np.zeros((phi_num, level_num - 3))

    a = tensor(destroy(N))
    phi = (a + a.dag()) * (8.0 * E_c / E_l) ** (0.25) / np.sqrt(2.0)
    na = 1.0j * (a.dag() - a) * (E_l / (8 * E_c)) ** (0.25) / np.sqrt(2.0)

    for idx, phi0 in enumerate(phi_ext):
        energies, states = ssj.bare_hamiltonian(N, E_l, E_c, E_j, phi0).eigenstates()
        for idy in range(level_num - 1):
            freq0x[idx, idy] = energies[idy + 1] - energies[0]
            phi0x[idx, idy] = abs(phi.matrix_element(states[0], states[idy + 1]))
            n0x[idx, idy] = abs(na.matrix_element(states[0], states[idy + 1]))
        for idy in range(level_num - 2):
            freq1x[idx, idy] = energies[idy + 2] - energies[1]
            phi1x[idx, idy] = abs(phi.matrix_element(states[1], states[idy + 2]))
            n1x[idx, idy] = abs(na.matrix_element(states[1], states[idy + 2]))
        for idy in range(level_num - 3):
            freq2x[idx, idy] = energies[idy + 3] - energies[2]
            phi2x[idx, idy] = abs(phi.matrix_element(states[2], states[idy + 3]))
            n2x[idx, idy] = abs(na.matrix_element(states[2], states[idy + 3]))
        printPercent(idx, phi_num)

    return [[freq0x, freq1x, freq2x], [phi0x, phi1x, phi2x], [n0x, n1x, n2x]]


def plotTransitions(I_array, freqList):
    level_num = freqList[0].shape[1] + 1
    [freq0x, freq1x, freq2x] = freqList
    for i in range(level_num - 1):
        plt.plot(I_array, freq0x[:, i], ':r')
    for i in range(level_num - 2):
        plt.plot(I_array, freq1x[:, i], ':g')
    for i in range(level_num - 3):
        plt.plot(I_array, freq2x[:, i], ':k')


def fitReflectionCircles(OneFreqUniqTrunc, OnePowerUniqTrunc, RComplexTrunc, guess, bounds):
    RForFit = RComplexTrunc.ravel()
    RForFit = np.concatenate((np.real(RForFit), np.imag(RForFit)))

    Power = 1e-3 * 10 ** (OnePowerUniqTrunc / 10)
    NumPowerTrunc = len(Power)
    opt, cov = curve_fit(reflection_for_fit, [OneFreqUniqTrunc, Power], RForFit, p0=guess, bounds=bounds, max_nfev=300000, ftol=1e-15)
    f0_fit, gamma_f_fit, P0_fit, A_fit, amp_cor_re_fit, amp_cor_im_fit, P0_im_fit= opt
    LargerFreqRange = np.linspace(f0_fit - gamma_f_fit * 10, f0_fit + gamma_f_fit * 10, 500)
    LargerFreqRange = np.concatenate(([0], LargerFreqRange, [0]))
    FittedComplex = reflection_for_fit([LargerFreqRange, Power], f0_fit, gamma_f_fit, P0_fit, A_fit, amp_cor_re_fit, amp_cor_im_fit, P0_im_fit)
    split_ind = int(len(FittedComplex) / 2)
    FittedComplex = FittedComplex[:split_ind] + 1j * FittedComplex[split_ind:]
    FittedComplex = FittedComplex.reshape((len(LargerFreqRange), NumPowerTrunc))
    return opt, cov, LargerFreqRange, FittedComplex


def reflection_for_fit(fPower, f0, gamma_f, P0, A, amp_cor_re, amp_cor_im, P0_im):
    f = fPower[0]
    num_f = len(f)
    Power = fPower[1]
    num_Power = len(Power)
    f = np.tensordot(f, np.ones(num_Power), axes=0)
    Power = np.tensordot(np.ones(num_f), Power, axes=0)
    r = (amp_cor_re + amp_cor_im * 1j) * (1 - 2 * (P0 + P0_im * 1j) * (1 + 2j * (f - f0) / gamma_f) / (1 + 4 * (f - f0) ** 2 / gamma_f ** 2 + A * Power))
    r = r.ravel()
    r = np.concatenate((np.real(r), np.imag(r)))  # for curve_fit
    # print('reflection_for_fit called')
    return r


def fitReflectionForCavityQ(Freq, RComplex, guess, bounds):
    RForFit = np.concatenate((np.real(RComplex), np.imag(RComplex)))
    opt, cov = curve_fit(reflection_for_Q_fit, Freq, RComplex, p0=guess, bounds=bounds, max_nfev=300000, ftol=1e-15)
    f0_fit, Qext_fit, Qint_fit = opt
    LargerFreqRange = Freq
    FittedComplex = reflection_for_Q_fit(Freq, )



def reflection_for_Q_fit(f, f0, Qext, Qint):
    r = (2j * (f - f0) / f0 - 1 / Qext + 1 / Qint) / ((2j * (f - f0) / f0 + 1 / Qext + 1 / Qint))
    return r


def printPercent(ind, tot):
    percent = ind / tot * 100
    if int(np.mod(percent, 7)) == 1:
        print('finish %d%%' % percent)