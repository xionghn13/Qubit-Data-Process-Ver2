from FunctionLibrary import *
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

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


def fit_gaussians_1D(x_data, y_data, num_gaussian=2, prominence_guess_factor=0.01,
 width_guess_factor=0.01):
    # only for positive gaussians
    num_pts = len(x_data)
    y_max = y_data.max()
    y_min = y_data.min()
    x_step = x_data[1] - x_data[0]
    y_range = y_max - y_min
    peaks, properties = find_peaks(y_data, prominence=y_range * prominence_guess_factor
    , width=num_pts * width_guess_factor)
    peak_heights = y_data[peaks] - y_min
    peak_widths = properties["widths"]
    peak_positions = x_data[peaks]
    if num_gaussian > len(peaks):
        raise ValueError('Only found %d peaks. Please change search criteria.' 
        % len(peaks))
    sort_height_index = np.argsort(peak_heights)
    sorted_heights = peak_heights[sort_height_index]
    sorted_widths = peak_widths[sort_height_index]
    sorted_positions = peak_positions[sort_height_index]
    height_guess = sorted_heights[-num_gaussian:]
    width_guess = sorted_widths[-num_gaussian:] * x_step
    position_guess = sorted_positions[-num_gaussian:]
    guess = np.concatenate([[y_min], height_guess, width_guess, position_guess])
    # print(guess)
    def multi_gaussian1D(x, *args):
        Z = args[0]
        parameter_matrix = np.reshape(args[1:], [3, num_gaussian])
        for i in range(num_gaussian):            
            Z += gaussian1D(x, 0, parameter_matrix[0, i],
            parameter_matrix[1, i], parameter_matrix[2, i])
        return Z
   

    try:
        opt, cov = curve_fit(multi_gaussian1D, x_data, y_data, guess)
    except RuntimeError:
        print("Error - curve_fit failed")
        opt = guess
        cov = np.zeros([len(opt), len(opt)])

    err = np.sqrt(np.diag(cov))
    # print(qopt)

    fit_x_data = np.linspace(x_data.min(), x_data.max(), 200)
    fit_y_data = multi_gaussian1D(fit_x_data, *opt)

    return opt, err, fit_x_data, fit_y_data


def fit_exponential(x_data, y_data, **kwargs):
    a_guess = y_data[-1]
    # y_data.mean() gives a very small number for centered data. That will cause an unexpected bug in curve_fit such
    # that the covariance matrix can't be calculated.
    b_guess = y_data[0] - a_guess
    T_guess = x_data[-1] / 5

    guess = [a_guess, b_guess, T_guess]
    # print(guess)
    function = exponential

    try:
        opt, cov = curve_fit(function, x_data, y_data, p0=guess, **kwargs)

    except RuntimeError:
        print("Error - curve_fit failed")
        opt = guess
        cov = np.zeros([len(opt), len(opt)])

    err = np.sqrt(cov.diagonal())
    fit_time = np.linspace(x_data.min(), x_data.max(), 200)
    fit_curve = function(fit_time, *opt)

    return opt, err, fit_time, fit_curve


def fit_ramsey(x_data, y_data, **kwargs):
    a_guess = (y_data.mean() + y_data[-1]) / 2
    # y_data.mean() gives a very small number for centered data. That will cause an unexpected bug in curve_fit such
    # that the covariance matrix can't be calculated.
    b_guess = (y_data.max() - y_data.min()) / 2 * np.sign(y_data[0] - a_guess)
    T_guess = x_data[-1]
    t0_guess = 0
    num_y = len(y_data)
    fourier = np.fft.fft(y_data - y_data.mean())
    time_step = x_data[1] - x_data[0]
    freq = np.fft.fftfreq(len(x_data), d=time_step)
    ind_max = np.argmax(np.abs(fourier))
    f_guess = np.abs(freq[ind_max])

    guess = [a_guess, b_guess, f_guess, t0_guess, T_guess]
    # print(guess)
    bound = [
    [-abs(abs(y_data).max() * 10), -abs(b_guess * 10), f_guess * 0.1, -x_data[-1], T_guess * 0.1],
    [abs(abs(y_data).max() * 10), abs(b_guess * 10), f_guess * 10, x_data[-1], T_guess * 10]
    ]
    # print(guess)
    function = damped_sinusoid

    try:
        opt, cov = curve_fit(function, x_data, y_data, p0=guess, bounds=bound, **kwargs)

    except RuntimeError:
        try:
            T_guess = x_data[-1] / 5
            guess = [a_guess, b_guess, f_guess, t0_guess, T_guess]
            bound = [
            [-abs(abs(y_data).max() * 10), -abs(b_guess * 10), f_guess * 0.1, -x_data[-1], T_guess * 0.1],
            [abs(abs(y_data).max() * 10), abs(b_guess * 10), f_guess * 10, x_data[-1], T_guess * 10]
            ]
            opt, cov = curve_fit(function, x_data, y_data, p0=guess, bounds=bound)
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

def fit_drifted_ramsey(x_data, y_data, **kwargs):
    a_guess = (y_data.mean() + y_data[-1]) / 2
    b_guess = (y_data.max() - y_data.min()) / 2
    c_guess = a_guess
    T1_guess = x_data[-1]
    T2_guess = T1_guess
    t0_guess = 0
    num_y = len(y_data)
    fourier = np.fft.fft(y_data - y_data.mean())
    time_step = x_data[1] - x_data[0]
    freq = np.fft.fftfreq(len(x_data), d=time_step)
    ind_max = np.argmax(np.abs(fourier))
    f_guess = np.abs(freq[ind_max])

    guess = [a_guess, b_guess, c_guess, f_guess, t0_guess, T1_guess, T2_guess]
    bound = [
    [-abs(abs(y_data).max() * 10), -abs(b_guess), -abs(abs(y_data).max() * 10),
    f_guess * 0.1, -x_data[-1], T1_guess * 0.1, T2_guess * 0.1],
    [abs(abs(y_data).max() * 10), abs(b_guess), abs(abs(y_data).max() * 10),
    f_guess * 10, x_data[-1], T1_guess * 10, T2_guess * 10]
    ]
    # print(guess)
    function = drifted_damped_sinusoid

    try:
        opt, cov = curve_fit(function, x_data, y_data, p0=guess, bounds=bound, **kwargs)

    except RuntimeError:
        print("Error - curve_fit failed")
        opt = guess
        cov = np.zeros([len(opt), len(opt)])

    err = np.sqrt(cov.diagonal())
    fit_time = np.linspace(x_data.min(), x_data.max(), 200)
    fit_curve = function(fit_time, *opt)

    return opt, err, fit_time, fit_curve
    
    
def fit_rabi(x_data, y_data, **kwargs):
    opt, err, fit_time, fit_curve = fit_ramsey(x_data, y_data, **kwargs)
    return opt, err, fit_time, fit_curve


def fit_RB(x_data, y_data, model=0, **kwargs):

    a_guess = y_data[-1]
    b_guess = y_data[0] - a_guess
    p_guess = 0.95

    if model == 0:
        guess = [p_guess, a_guess, b_guess]
        # print(guess)
        function = randomized_benchmarking_0
    elif model == 1:
        guess = [p_guess, a_guess, b_guess, 0.035 * b_guess]
        # print(guess)
        function = randomized_benchmarking_1
    else:
        raise ValueError('model = %s? No such fucking model!' % model)
    try:
        opt, cov = curve_fit(function, x_data, y_data, p0=guess, **kwargs)

    except RuntimeError:
        print("Error - curve_fit failed")
        opt = guess
        cov = np.zeros([len(opt), len(opt)])

    err = np.sqrt(cov.diagonal())
    fit_time = np.linspace(x_data.min(), x_data.max(), 200)
    fit_curve = function(fit_time, *opt)

    return opt, err, fit_time, fit_curve

def fit_XEB(x_data, y_data, **kwargs):

    b_guess = y_data[0]
    p_guess = 0.95

    def XEB_function(m, p, b):
        return randomized_benchmarking_0(m, p, 0, b)

    guess = [p_guess, b_guess]
    # print(guess)
    function = XEB_function

    try:
        opt, cov = curve_fit(function, x_data, y_data, p0=guess, **kwargs)

    except RuntimeError:
        print("Error - curve_fit failed")
        opt = guess
        cov = np.zeros([len(opt), len(opt)])

    err = np.sqrt(cov.diagonal())
    fit_time = np.linspace(x_data.min(), x_data.max(), 200)
    fit_curve = function(fit_time, *opt)

    return opt, err, fit_time, fit_curve

def fit_multi_RB(x_data, y_data, model=0, **kwargs):

    a_guess = y_data[0, -1]
    b_guess = y_data[0, 0] - a_guess
    p_guess = 0.95
    num_measurements = y_data.shape[0]
    num_points = y_data.shape[1]
    
    guess = list(np.zeros(num_measurements) + p_guess)

    if model == 0:
        guess += [a_guess, b_guess]
        # print(guess)
        function = randomized_benchmarking_0
    elif model == 1:
        guess += [a_guess, b_guess, 0.035 * b_guess]
        # print(guess)
        function = randomized_benchmarking_1
    else:
        raise ValueError('model = %s? No such fucking model!' % model)
        
    def func(m, *args):
        arg_list = list(args)
        output = np.array([])
        for i in range(num_measurements):
            argi = [arg_list[i]] + arg_list[num_measurements:]
            output = np.concatenate([output, function(m, *argi)])
        return output

    try:
        opt, cov = curve_fit(func, x_data, y_data.ravel(), p0=guess, **kwargs)

    except RuntimeError:
        print("Error - curve_fit failed")
        opt = guess
        cov = np.zeros([len(opt), len(opt)])

    err = np.sqrt(cov.diagonal())
    fit_time = np.linspace(x_data.min(), x_data.max(), 200)
    fit_curve = np.reshape(func(fit_time, *opt), (num_measurements, 200))

    return opt, err, fit_time, fit_curve
    

def fit_dephasing_and_frequency_shift_with_cavity_photons(x_data, dephasing_data, frequency_shift_data,
                                                          method='fix_Gamma_2_kappa_omega_c_omega_0', Gamma_2=0,
                                                          kappa=0, omega_0=0, omega_c=0, **kwargs):
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
        opt, cov = curve_fit(function, x_data, y_data, p0=guess, **kwargs)

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


def fitReflectionCircles(OneFreqUniqTrunc, OnePowerUniqTrunc, RComplexTrunc, guess, bounds, **kwargs):
    RForFit = RComplexTrunc.ravel()
    RForFit = np.concatenate((np.real(RForFit), np.imag(RForFit)))

    Power = 1e-3 * 10 ** (OnePowerUniqTrunc / 10)
    NumPowerTrunc = len(Power)
    opt, cov = curve_fit(reflection_for_fit, [OneFreqUniqTrunc, Power], RForFit, p0=guess, bounds=bounds,
                         max_nfev=300000, ftol=1e-15, **kwargs)
    f0_fit, gamma_f_fit, P0_fit, A_fit, amp_cor_re_fit, amp_cor_im_fit, P0_im_fit = opt
    LargerFreqRange = np.linspace(f0_fit - gamma_f_fit * 10, f0_fit + gamma_f_fit * 10, 500)
    LargerFreqRange = np.concatenate(([0], LargerFreqRange, [0]))
    FittedComplex = reflection_for_fit([LargerFreqRange, Power], f0_fit, gamma_f_fit, P0_fit, A_fit, amp_cor_re_fit,
                                       amp_cor_im_fit, P0_im_fit)
    split_ind = int(len(FittedComplex) / 2)
    FittedComplex = FittedComplex[:split_ind] + 1j * FittedComplex[split_ind:]
    FittedComplex = FittedComplex.reshape((len(LargerFreqRange), NumPowerTrunc))
    return opt, cov, LargerFreqRange, FittedComplex


def reflection_for_fit(fPower, f0, gamma_f, P0, A, amp_cor_re, amp_cor_im, P0_im, **kwargs):
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
