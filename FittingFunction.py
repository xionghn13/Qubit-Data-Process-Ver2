from FunctionLibrary import *
from scipy.optimize import curve_fit


def fit_transient_time(DrivePowerArray, Gamma_r, OptMatrix, power_for_plot=[]):
    if len(power_for_plot) == 0:
        power_for_plot = DrivePowerArray
    x_data = [DrivePowerArray, Gamma_r]
    x_for_plot = [power_for_plot, Gamma_r]
    y_data = OptMatrix[1, :] / 1000
    pow_mean = 1e-3 * 10 ** (x_data[0].mean() / 10)
    pow_ratio_guess = Gamma_r ** 2 / 2 / pow_mean
    Gamma_in_guess = 1 / y_data.max()
    Gamma_out_guess = 1 / y_data.min()
    guess = ([Gamma_in_guess, Gamma_out_guess, pow_ratio_guess])
    bounds = (
        (0, 0, 0),
        (10 * Gamma_r, 10 * Gamma_r, np.inf)
    )
    opt, cov = curve_fit(transient_time_curve, x_data, y_data, p0=guess, bounds=bounds)
    Gamma_in_fit, Gamma_out_fit, pow_ratio_fit = opt
    FitTime = transient_time_curve(x_for_plot, Gamma_in_fit, Gamma_out_fit, pow_ratio_fit)
    return opt, cov, FitTime


def fit_lorentzian(frequency, V_sq_abs):
    MaxAbs_0 = np.max(V_sq_abs)
    V_sq_abs_fit = V_sq_abs / MaxAbs_0
    # V_sq_abs_fit = V_sq_abs
    MaxAbs = np.max(V_sq_abs_fit)
    MinAbs = np.min(V_sq_abs_fit)
    MaxInd = V_sq_abs_fit.argmax()
    # print(MaxInd)
    f0_guess = frequency[MaxInd]
    kappa_guess = (frequency[-1] - frequency[0]) / 4
    b_guess = MinAbs
    a_guess = (MaxAbs - MinAbs) * (kappa_guess / 2) ** 2

    guess = [a_guess, b_guess, f0_guess, kappa_guess]
    # print(guess)
    bounds = (
        (0, 0, frequency[0], 0),
        (MaxAbs * 10, MaxAbs, frequency[-1], kappa_guess * 4)
    )

    try:
        opt, cov = curve_fit(lorentzian, frequency, V_sq_abs_fit, guess, bounds=bounds)
    except RuntimeError:
        print("Error - curve_fit failed")
        opt = guess
        cov = np.zeros([len(opt), len(opt)])

    err = np.sqrt(np.diag(cov))
    opt[0:2] *= MaxAbs_0
    err[0:2] *= MaxAbs_0
    # print(qopt)

    fit_x_data = np.linspace(frequency.min(), frequency.max(), 200)
    fit_abs = lorentzian(fit_x_data, *opt)

    return opt, err, fit_x_data, fit_abs


def fit_exponential(x_data, y_data):
    a_guess = y_data[-1]
    # y_data.mean() gives a very small number for centered data. That will cause an unexpected bug in curve_fit such
    # that the covariance matrix can't be calculated.
    b_guess = y_data[0] - a_guess
    T_guess = x_data[-1] / 5

    guess = [a_guess, b_guess, T_guess]
    # print(guess)
    function = exponential

    try:
        opt, cov = curve_fit(function, x_data, y_data, p0=guess)

    except RuntimeError:
        print("Error - curve_fit failed")
        opt = guess
        cov = np.zeros([len(opt), len(opt)])

    err = np.sqrt(cov.diagonal())
    fit_time = np.linspace(x_data.min(), x_data.max(), 200)
    fit_curve = function(fit_time, *opt)

    return opt, err, fit_time, fit_curve


def fit_ramsey(x_data, y_data):
    a_guess = (y_data.mean() + y_data[-1]) / 2
    # y_data.mean() gives a very small number for centered data. That will cause an unexpected bug in curve_fit such
    # that the covariance matrix can't be calculated.
    b_guess = y_data[0] - a_guess
    T_guess = x_data[-1] / 5
    t0_guess = 0

    fourier = np.fft.fft(y_data - y_data.mean())
    time_step = x_data[1] - x_data[0]
    freq = np.fft.fftfreq(len(x_data), d=time_step)
    ind_max = np.argmax(np.abs(fourier))
    f_guess = freq[ind_max]

    guess = [a_guess, b_guess, f_guess, t0_guess, T_guess]
    # print(guess)
    function = damped_sinusoid

    try:
        opt, cov = curve_fit(function, x_data, y_data, p0=guess)

    except RuntimeError:
        try:
            T_guess = x_data[-1]
            guess = [a_guess, b_guess, f_guess, t0_guess, T_guess]
            opt, cov = curve_fit(function, x_data, y_data, p0=guess)
        except RuntimeError:   
            try:
                opt_exp, err_exp, fit_time, fit_curve = fit_exponential(x_data, y_data)
                opt = np.zeros(5,) + np.nan
                err = np.zeros(5,) + np.nan
                opt[[0, 1, -1]] = opt_exp
                err[[0, 1, -1]] = err_exp
                return opt, err, fit_time, fit_curve
            except RuntimeError:
                print("Error - curve_fit failed")
                opt = guess
                cov = np.zeros([len(opt), len(opt)])

    err = np.sqrt(cov.diagonal())
    fit_time = np.linspace(x_data.min(), x_data.max(), 200)
    fit_curve = function(fit_time, *opt)

    return opt, err, fit_time, fit_curve


def fit_rabi(x_data, y_data):
    opt, err, fit_time, fit_curve = fit_ramsey(x_data, y_data)
    return opt, err, fit_time, fit_curve


def fit_dephasing_and_frequency_shift_with_cavity_photons(x_data, dephasing_data, frequency_shift_data,
                                                          method='fix_Gamma_2_kappa_omega_c_omega_0', Gamma_2=0,
                                                          kappa=0, omega_0=0, omega_c=0):
    y_data = np.concatenate((dephasing_data, frequency_shift_data))

    if method == 'fix_Gamma_2_kappa_omega_c_omega_0':
        chi_guess = kappa
        max_Gamma_d = np.max(dephasing_data) - np.min(dephasing_data)
        epsilon_guess = np.sqrt(
            max_Gamma_d / 8 / kappa / chi_guess ** 2 * (
                    (kappa ** 2 - chi_guess ** 2) ** 2 + 4 * chi_guess ** 2 * kappa ** 2
            )
        )

        guess = [chi_guess, epsilon_guess]

        def function(omega_d, chi, epsilon):
            Gamma_fit = dephasing_with_cavity_photons(omega_d, chi, epsilon, Gamma_2, kappa, omega_c)
            omega_fit = frequency_shift_with_cavity_photons(omega_d, chi, epsilon, kappa, omega_0, omega_c)
            return np.concatenate((Gamma_fit, omega_fit))

    elif method == 'fix_Gamma_2_kappa_omega_0':
        chi_guess = kappa / 10
        max_Gamma_d = np.max(dephasing_data) - np.min(dephasing_data)
        epsilon_guess = np.sqrt(
            max_Gamma_d / 8 / kappa / chi_guess ** 2 * (
                    (kappa ** 2 - chi_guess ** 2) ** 2 + 4 * chi_guess ** 2 * kappa ** 2
            )
        )
        omega_c_guess = np.mean(x_data)
        guess = [chi_guess, epsilon_guess, omega_c_guess]

        def function(omega_d, chi, epsilon, omega_c):
            Gamma_fit = dephasing_with_cavity_photons(omega_d, chi, epsilon, Gamma_2, kappa, omega_c)
            omega_fit = frequency_shift_with_cavity_photons(omega_d, chi, epsilon, kappa, omega_0, omega_c)
            return np.concatenate((Gamma_fit, omega_fit))

    elif method == 'fix_Gamma_2_omega_0':
        kappa_guess = (np.max(x_data) - np.min(x_data)) / 5
        chi_guess = kappa_guess / 10
        max_Gamma_d = np.max(dephasing_data) - np.min(dephasing_data)
        epsilon_guess = np.sqrt(
            max_Gamma_d / 8 / kappa_guess / chi_guess ** 2 * (
                    (kappa_guess ** 2 - chi_guess ** 2) ** 2 + 4 * chi_guess ** 2 * kappa_guess ** 2
            )
        )
        omega_c_guess = np.mean(x_data)
        guess = [chi_guess, epsilon_guess, kappa_guess, omega_c_guess]

        def function(omega_d, chi, epsilon, kappa, omega_c):
            Gamma_fit = dephasing_with_cavity_photons(omega_d, chi, epsilon, Gamma_2, kappa, omega_c)
            omega_fit = frequency_shift_with_cavity_photons(omega_d, chi, epsilon, kappa, omega_0, omega_c)
            return np.concatenate((Gamma_fit, omega_fit))
    elif method == 'fix_kappa':
        Gamma_2_guess = dephasing_data[0]
        omega_0_guess = frequency_shift_data[0]
        chi_guess = kappa / 10
        max_Gamma_d = np.max(dephasing_data) - np.min(dephasing_data)
        epsilon_guess = np.sqrt(
            max_Gamma_d / 8 / kappa / chi_guess ** 2 * (
                    (kappa ** 2 - chi_guess ** 2) ** 2 + 4 * chi_guess ** 2 * kappa ** 2
            )
        )
        omega_c_guess = np.mean(x_data)
        guess = [chi_guess, epsilon_guess, Gamma_2_guess, omega_0_guess, omega_c_guess]

        def function(omega_d, chi, epsilon, Gamma_2, omega_0, omega_c):
            Gamma_fit = dephasing_with_cavity_photons(omega_d, chi, epsilon, Gamma_2, kappa, omega_c)
            omega_fit = frequency_shift_with_cavity_photons(omega_d, chi, epsilon, kappa, omega_0, omega_c)
            return np.concatenate((Gamma_fit, omega_fit))
    elif method == 'fix_no_parameters':
        Gamma_2_guess = dephasing_data[0]
        omega_0_guess = frequency_shift_data[0]
        kappa_guess = (np.max(x_data) - np.min(x_data)) / 5
        chi_guess = kappa_guess / 10
        max_Gamma_d = np.max(dephasing_data) - np.min(dephasing_data)
        epsilon_guess = np.sqrt(
            max_Gamma_d / 8 / kappa_guess / chi_guess ** 2 * (
                    (kappa_guess ** 2 - chi_guess ** 2) ** 2 + 4 * chi_guess ** 2 * kappa_guess ** 2
            )
        )
        omega_c_guess = np.mean(x_data)
        guess = [chi_guess, epsilon_guess, Gamma_2_guess, kappa_guess, omega_0_guess, omega_c_guess]

        def function(omega_d, chi, epsilon, Gamma_2, kappa, omega_0, omega_c):
            Gamma_fit = dephasing_with_cavity_photons(omega_d, chi, epsilon, Gamma_2, kappa, omega_c)
            omega_fit = frequency_shift_with_cavity_photons(omega_d, chi, epsilon, kappa, omega_0, omega_c)
            return np.concatenate((Gamma_fit, omega_fit))
    else:
        raise ValueError('method = %s? No such fucking method!' % method)

    try:
        opt, cov = curve_fit(function, x_data, y_data, p0=guess)

    except RuntimeError:
        print("Error - curve_fit failed")
        opt = guess
        cov = np.zeros([len(opt), len(opt)])

    err = np.sqrt(cov.diagonal())
    n_pts = 200
    fit_x = np.linspace(x_data.min(), x_data.max(), n_pts)
    fit_curve = function(fit_x, *opt)
    Gamma_fit = fit_curve[:n_pts]
    omega_fit = fit_curve[n_pts:]
    return opt, err, fit_x, Gamma_fit, omega_fit


def fitReflectionCircles(OneFreqUniqTrunc, OnePowerUniqTrunc, RComplexTrunc, guess, bounds):
    RForFit = RComplexTrunc.ravel()
    RForFit = np.concatenate((np.real(RForFit), np.imag(RForFit)))

    Power = 1e-3 * 10 ** (OnePowerUniqTrunc / 10)
    NumPowerTrunc = len(Power)
    opt, cov = curve_fit(reflection_for_fit, [OneFreqUniqTrunc, Power], RForFit, p0=guess, bounds=bounds,
                         max_nfev=300000, ftol=1e-15)
    f0_fit, gamma_f_fit, P0_fit, A_fit, amp_cor_re_fit, amp_cor_im_fit, P0_im_fit = opt
    LargerFreqRange = np.linspace(f0_fit - gamma_f_fit * 10, f0_fit + gamma_f_fit * 10, 500)
    LargerFreqRange = np.concatenate(([0], LargerFreqRange, [0]))
    FittedComplex = reflection_for_fit([LargerFreqRange, Power], f0_fit, gamma_f_fit, P0_fit, A_fit, amp_cor_re_fit,
                                       amp_cor_im_fit, P0_im_fit)
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
    r = (amp_cor_re + amp_cor_im * 1j) * (1 - 2 * (P0 + P0_im * 1j) * (1 + 2j * (f - f0) / gamma_f) / (
                1 + 4 * (f - f0) ** 2 / gamma_f ** 2 + A * Power))
    r = r.ravel()
    r = np.concatenate((np.real(r), np.imag(r)))  # for curve_fit
    # print('reflection_for_fit called')
    return r
