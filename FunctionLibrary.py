import numpy as np
from enum import Enum


def cosine(t, a, b, f):
    return a + b * np.cos( 2 * np.pi * f * t)

def damped_cosine(t, a, b, f, T):
    return a + b * np.cos( 2 * np.pi * f * t) * np.exp(- t / T)
    
def sinusoid(t, a, b, f, t0):
    return a + b * np.cos( 2 * np.pi * f * (t-t0))
    
def damped_sinusoid(t, a, b, f, t0,T):
    return a + b * np.cos( 2 * np.pi * f * (t-t0)) * np.exp(- t / T)
    
def randomized_benchmarking_0(m, p, a, b):
    return a + b * p ** m

def randomized_benchmarking_1(m, p, a, b, c):
    return a + b * p ** m + c * (m - 1) * p ** (m - 2)

def exponential(t, a, b, T):
    return a + b * np.exp(-t/T)
    
def two_exponential(t, a, b, T1, T2):
    return a + b * np.exp(-t/T1) + c * np.exp(-t/T2)
    
def double_exponential(t, a, b, c, T1, Lambda, T1qp):
    return a + b * np.exp(- t / T1 + Lambda * (np.exp(- t / T1qp) - 1))

def lorentzian(f, a, b, f0, kappa):
    return a / ((f-f0)**2 + (kappa/2)**2) + b

def gaussian1D(x, a, b, sigma, x0):
    return a + b * np.exp(- (x-x0)**2 / (2 * sigma**2))
   
def gaussian2D(x, y, a, b, sigmax, sigmay, x0, y0):
    return a + b * np.exp(- ((x-x0)**2 /(2 * sigmax**2) + (y-y0)**2) / (2 * sigmay**2) )

#Old Stuff

# def T1_curve(t, A, T1, B):
    # r = A * np.exp(- t / T1) + B
    #print('T1 = %.3Gus, A = %.3G, B = %.3G' % (T1 / 1000, A, B))
    # return r


# def DoubleExp_curve(t, A, TR, B, Tqp, lamb):
    # r = A * np.exp(- t / TR + lamb * (np.exp(- t / Tqp) - 1)) + B
    #print('T1 = %.3Gus, A = %.3G, B = %.3G' % (T1 / 1000, A, B))
    # return r


# def TwoExp_curve(t, A, T1, B, T2, C):
    # r = A * np.exp(- t / T1) + B * np.exp(- t / T2) + C
    # return r


# def rabi_curve(t, A, T1, B, Tpi, phi0):
    # r = A * np.exp(- t / T1) * np.cos(t / Tpi * np.pi + phi0) + B
    # return r


# def rabi_two_exp_curve(t, A, T1, B, Tpi, phi0, T_out, C):
    # r = A * np.exp(- t / T1) * np.cos(t / Tpi * np.pi + phi0) + B * np.exp(-t / T_out) + C
    # return r


# def transient_time_curve(power_dBm_Gamma_r, Gamma_in, Gamma_out, A):
    # power_dBm = power_dBm_Gamma_r[0]
    # Gamma_r = power_dBm_Gamma_r[1]
    # power = 1e-3 * 10 ** (power_dBm / 10)
    # Gamma = Gamma_in + power * A * Gamma_out / (Gamma_r ** 2 + 2 * power * A)
    # time = 1 / Gamma
    # return time


# def lorenztian(f, f0, kappa, A, B):
    # t = A / ((f - f0) ** 2 + (kappa / 2) ** 2) + B
    # return t


# move FitTransientTime to FittingFunc


# move AutoRotate to DataManipulationFunc


