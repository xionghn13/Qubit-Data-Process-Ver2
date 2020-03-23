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


def fit_lorenztian(frequency, V_sq_abs):
    MaxAbs_0 = np.max(V_sq_abs)
    V_sq_abs_fit = V_sq_abs / MaxAbs_0
    # V_sq_abs_fit = V_sq_abs
    MaxAbs = np.max(V_sq_abs_fit)
    MinAbs = np.min(V_sq_abs_fit)
    MaxInd = V_sq_abs_fit.argmax()
    # print(MaxInd)
    f0_guess = frequency[MaxInd]
    kappa_guess = (frequency[-1] - frequency[0]) / 4
    B_guess = MinAbs
    A_guess = (MaxAbs - MinAbs) * (kappa_guess / 2) ** 2

    guess = ([f0_guess, kappa_guess, A_guess, B_guess])
    # print(guess)
    bounds = (
        (frequency[0], 0, 0, 0),
        (frequency[-1], kappa_guess * 4, MaxAbs * 10, MaxAbs)
    )

    opt, cov = curve_fit(lorenztian, frequency, V_sq_abs_fit, guess, bounds=bounds)
    # f0_fit, kappa_fit, A_fit, B_fit = qopt
    # print(qopt)
    err = np.sqrt(np.diag(cov))
    opt[2:] *= MaxAbs_0
    err[2:] *= MaxAbs_0
    # print(qopt)
    fit_abs = lorenztian(frequency, *opt)

    return opt, err, fit_abs

def fit_rabi(x_data, y_data):
    B_guess = y_data.mean()
    A_guess = y_data[0] - B_guess
    T1_guess = x_data[-1]

    Tpi_guess = T1_guess / 4
    phi0_guess = 0
    guess = ([A_guess, T1_guess, B_guess, Tpi_guess, phi0_guess])
    bounds = (
        (-np.inf, 0, -np.inf, 0, - np.pi / 2),
        (np.inf, np.inf, np.inf, np.inf, np.pi / 2)
    )

    try:
        opt, cov = curve_fit(rabi_curve, x_data, y_data, p0=guess, bounds=bounds)
    except RuntimeError:
        print("Error - curve_fit failed")
        opt = guess
        cov = np.zeros([len(opt), len(opt)])
    A_fit, T1_fit, B_fit, Tpi_fit, phi0_fit = opt
    err = np.sqrt(cov.diagonal())
    fit_time = np.linspace(x_data.min(), x_data.max(), 200)
    fit_curve = rabi_curve(fit_time, A_fit, T1_fit, B_fit, Tpi_fit, phi0_fit)

    return opt, err, fit_time, fit_curve