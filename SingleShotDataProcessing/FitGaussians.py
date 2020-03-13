import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x, y: height * np.exp(
        -(((center_x - x) / width_x) ** 2 + ((center_y - y) / width_y) ** 2) / 2)

def multi_gaussian(X, Y, param_mat):
    n_gaussian = param_mat.shape[0]
    Z = 0
    for i in range(n_gaussian):
        # print(param_mat[i, :])
        Z += gaussian(*param_mat[i, :])(X, Y)
    return Z

# def moments(data):
#     """Returns (height, x, y, width_x, width_y)
#     the gaussian parameters of a 2D distribution by calculating its
#     moments """
#     total = data.sum()
#     X, Y = np.indices(data.shape)
#     x = (X * data).sum() / total
#     y = (Y * data).sum() / total
#     col = data[:, int(y)]
#     width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
#     row = data[int(x), :]
#     width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
#     height = data.max()
#     return height, x, y, width_x, width_y


def fit_gaussian(X, Y, data, param_mat, method='same_width', fixed_parameters=[]):
    """Returns the gaussian parameters of a 2D distribution found by a fit"""
    # print(param_mat.shape)
    n_gaussian = len(param_mat)
    if method == 'same_width':
        p = np.ones_like(param_mat)
        p_err = np.ones_like(param_mat)
        param = np.append(param_mat[:, :3].ravel(), param_mat[0, 3])
        def error_func(param):
            param_matrix = np.ones((n_gaussian, 5)) * param[-1]
            param_matrix[:, :3] = np.reshape(param[:-1], (n_gaussian, 3))
            data_sim = multi_gaussian(X, Y, param_matrix)
            return np.ravel((data_sim - data) ** 2)
        res = optimize.leastsq(error_func, param, full_output=1)
        p *= res[0][-1]
        p[:, :3] = np.reshape(res[0][:-1], (n_gaussian, 3))
        p_cov = res[1]
        err = np.sqrt(p_cov.diagonal())
        p_err *= err[-1]
        p_err[:, :3] = np.reshape(err[:-1], (n_gaussian, 3))
    elif method == 'different_width':    
        param = param_mat.ravel()
        def error_func(param):
            param_matrix = np.reshape(param, (n_gaussian, 5))
            data_sim = multi_gaussian(X, Y, param_matrix)
            return np.ravel((data_sim - data) ** 2)
        res = optimize.leastsq(error_func, param, full_output=1)
        p = res[0]
        p_cov = res[1]
        p = np.reshape(p, (n_gaussian, 5))
        p_err = np.sqrt(p_cov.diagonal())
        p_err = np.reshape(p_err, (n_gaussian, 5))
    elif method == 'fix_positions_and_widths':
        param = param_mat.ravel()
        def error_func(param):
            param_matrix = np.reshape(param, (n_gaussian, 1))
            param_matrix = np.concatenate((param_matrix, fixed_parameters), axis=1)
            data_sim = multi_gaussian(X, Y, param_matrix)
            return np.ravel((data_sim - data) ** 2)
        res = optimize.leastsq(error_func, param, full_output=1)
        p = res[0]
        p_cov = res[1]
        p_err = np.sqrt(p_cov.diagonal())
    else:
        raise ValueError('method = %s? No such fucking method!' % method)
    return p, p_err

if __name__ == '__main__':
    # Create the gaussian data
    Xin, Yin = np.mgrid[0:301, 0:301]
    data = gaussian(3, 50, 50, 10, 10)(Xin, Yin) \
           + gaussian(3, 50, 250, 10, 10)(Xin, Yin) \
           + gaussian(3, 250, 50, 10, 10)(Xin, Yin) \
           + gaussian(3, 280, 280, 10, 10)(Xin, Yin) \
           + np.random.random(Xin.shape)

    plt.matshow(data, cmap=plt.cm.gist_earth_r)
    print(Xin.shape)
    print(data.shape)
    height_guess = np.max(data)
    print(height_guess)
    width_x_guess = (np.max(Xin) - np.min(Xin)) / 10
    width_y_guess = width_x_guess
    center_guess = [[40, 40],
                    [40, 260],
                    [260, 40],
                    [280, 280]]
    param_list = []
    for pt in center_guess:
        param_list += [[height_guess] + pt + [width_x_guess, width_y_guess]]
    param_mat = np.array(param_list)

    params, params_err = fit_gaussian(Xin, Yin, data, param_mat)
    print('-----')
    print(params)
    print(params_err)

    fit = multi_gaussian(Xin, Yin, params)

    plt.contour(fit, cmap=plt.cm.copper)
    ax = plt.gca()
    (height, x, y, width_x, width_y) = params[0, :]

    plt.text(0.95, 0.05, """
    x : %.1f
    y : %.1f
    width_x : %.1f
    width_y : %.1f""" % (x, y, width_x, width_y),
             fontsize=16, horizontalalignment='right',
             verticalalignment='bottom', transform=ax.transAxes)
    plt.show()
