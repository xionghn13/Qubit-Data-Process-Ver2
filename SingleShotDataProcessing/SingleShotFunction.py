import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
import SingleShotDataProcessing.FitGaussians as fg


def get_blob_centers(x, y, num_blob):
    num_points = len(x)
    matrix = np.array([x, y]).transpose()
    if num_points > 10e3:
        kmeans = MiniBatchKMeans(n_clusters=num_blob).fit(matrix)
    else:
        kmeans = KMeans(n_clusters=num_blob).fit(matrix)
    centers = kmeans.cluster_centers_
    num_batch = min(10000, num_points)
    labels = kmeans.labels_[:num_batch]
    x_batch = x[:num_batch]
    y_batch = y[:num_batch]
    sigmas = np.zeros_like(centers)
    for i in range(len(centers)):
        sigmas[i, 0] = np.std(x_batch[labels == i])
        sigmas[i, 1] = np.std(y_batch[labels == i])
    return centers, sigmas


def get_center_heights(X, Y, H, centers):
    num_blob = len(centers)
    heights = np.zeros((num_blob,))
    for i in range(num_blob):
        x_c = centers[i, 0]
        y_c = centers[i, 1]
        ind_x = np.argmin((np.unique(X) - x_c) ** 2)
        ind_y = np.argmin((np.unique(Y) - y_c) ** 2)
        heights[i] = H[ind_y, ind_x]

    return heights


def complex_array_to_2d_histogram(complex_array, bin=100):
    sReal = np.real(complex_array)
    sImag = np.imag(complex_array)
    H, xedges, yedges = np.histogram2d(sReal, sImag, bins=[bin, bin])
    X, Y = np.meshgrid(xedges[1:], yedges[1:])
    H = H.T
    return X, Y, H


def full_blob_analysis(complex_array, bin=100, num_blob=4, method='same_width'):
    sReal = np.real(complex_array)
    sImag = np.imag(complex_array)
    X, Y, H = complex_array_to_2d_histogram(complex_array, bin=bin)
    centers, sigmas = get_blob_centers(sReal, sImag, num_blob)
    heights = get_center_heights(X, Y, H, centers)
    param_mat = np.concatenate((heights.reshape(num_blob, 1), centers, sigmas), axis=1)
    params, params_err = fg.fit_gaussian(X, Y, H, param_mat, method=method)
    heights_fit, centers_fit, sigmas_fit = unwrap_blob_parameters(params)
    trial = 0
    while np.any(heights_fit < 0) and trial < 0:
        centers, sigmas = get_blob_centers(sReal, sImag, num_blob)
        heights = get_center_heights(X, Y, H, centers)
        param_mat = np.concatenate((heights.reshape(num_blob, 1), centers, sigmas), axis=1)
        params, params_err = fg.fit_gaussian(X, Y, H, param_mat, method=method)
        heights_fit, centers_fit, sigmas_fit = unwrap_blob_parameters(params)
        trial += 1
    return params, params_err


def fit_blob_height(complex_array, centers, sigmas, bin=100):
    sReal = np.real(complex_array)
    sImag = np.imag(complex_array)
    X, Y, H = complex_array_to_2d_histogram(complex_array, bin=bin)
    heights = get_center_heights(X, Y, H, centers)
    fixed_param = np.concatenate((centers, sigmas), axis=1)
    params, params_err = fg.fit_gaussian(X, Y, H, heights, method='fix_positions_and_widths',
                                         fixed_parameters=fixed_param)
    return params, params_err


def count_points_in_blob(complex_array, centers, sigmas, width_threshold=2):
    num_blob = centers.shape[0]
    counts = np.zeros((num_blob,))
    for i in range(num_blob):
        data_index = data_point_index_for_blob(complex_array, centers[i, :], sigmas[i, :], width_threshold=width_threshold)
        counts[i] = np.sum(data_index)
    return counts


def unwrap_blob_parameters(params):
    centers_fit = params[:, 1:3]
    sigmas_fit = params[:, 3:]
    heights_fit = params[:, 0]
    return heights_fit, centers_fit, sigmas_fit


def closest_blob_index(centers, position_estimate):
    return np.argmin(np.sum((centers - position_estimate) ** 2, axis=1))


def data_point_index_for_blob(heralding_signal, blob_center, blob_sigma, width_threshold=2):
    # data index is in the form of boolean array
    sReal = heralding_signal.real
    sImag = heralding_signal.imag
    data_index = ((sReal - blob_center[0]) / blob_sigma[0] / width_threshold) ** 2 + (
            (sImag - blob_center[1]) / blob_sigma[1] / width_threshold) ** 2 < 1
    return data_index


def blob_population_to_qubit_population(population_matrix, label_list):
    pgA = population_matrix[label_list.index('gg')] + population_matrix[label_list.index('ge')]
    peA = population_matrix[label_list.index('eg')] + population_matrix[label_list.index('ee')]

    pgB = population_matrix[label_list.index('gg')] + population_matrix[label_list.index('eg')]
    peB = population_matrix[label_list.index('ge')] + population_matrix[label_list.index('ee')]
    return pgA, peA, pgB, peB


def blob_population_std_to_qubit_population_std(population_matrix_std, label_list):
    pgA_std = np.sqrt(
        population_matrix_std[label_list.index('gg')] ** 2 + population_matrix_std[label_list.index('ge')] ** 2)
    peA_std = np.sqrt(
        population_matrix_std[label_list.index('eg')] ** 2 + population_matrix_std[label_list.index('ee')] ** 2)

    pgB_std = np.sqrt(
        population_matrix_std[label_list.index('gg')] ** 2 + population_matrix_std[label_list.index('eg')] ** 2)
    peB_std = np.sqrt(
        population_matrix_std[label_list.index('ge')] ** 2 + population_matrix_std[label_list.index('ee')] ** 2)
    return pgA_std, peA_std, pgB_std, peB_std


def p_to_fidelity(p_arr, p_err_arr, dim):
    parameter = np.average(p_arr)
    num_p = len(p_arr)
    parameter_err = np.sqrt(np.sum(p_err_arr ** 2)) / num_p
    error = abs((dim - 1) * (1 - parameter) / dim)
    error_err = (dim - 1) * parameter_err / dim
    error = error
    error_err = error_err
    return [parameter, parameter_err, error, error_err]


def individual_p_to_fidelity(p_arr, p_err_arr, dim, label_list):
    for i in range(len(p_arr)):
        parameter = p_arr[i]
        parameter_err = p_err_arr[i]
        error = abs((dim - 1) * (1 - parameter) / dim)
        error_err = (dim - 1) * parameter_err / dim
        print('for blob %s, p=%.4G\u00B1%.4G, fidelity %.4G\u00B1%.4G' % (
            label_list[i], parameter, parameter_err, 1 - error, error_err))


def p_to_fidelity_interleaved(p_arr, p_err_arr, dim, p, p_err, blob_ind):
    if type(blob_ind) is list:
        parameter_0 = np.average(p_arr[blob_ind])
        num_p = len(p_arr)
        parameter_0_err = np.sqrt(np.sum(p_err_arr[blob_ind] ** 2)) / num_p
    else:
        parameter_0 = p_arr[blob_ind]
        parameter_0_err = p_err_arr[blob_ind]

    parameter = parameter_0 / p
    parameter_err = parameter * np.sqrt((parameter_0_err / parameter_0) ** 2 + (p_err / p) ** 2)
    error = abs((dim - 1) * (1 - parameter) / dim)
    error_err = (dim - 1) * parameter_err / dim

    return [parameter, parameter_err, error, error_err]


def cross_entropy(p, q):
    return -np.sum(p * np.log(q))

def XEB_sequence_fidelity(p_measured, p_expected):
    # the length of p_measured should be 2 ** n
    p_incoherent = np.zeros_like(p_measured)
    p_incoherent = p_incoherent + 1. / len(p_incoherent)
    F_XEB = (cross_entropy(p_incoherent, p_expected) - cross_entropy(
        p_measured, p_expected)) / (cross_entropy(p_incoherent, p_expected) - cross_entropy(p_expected, p_expected))
    return F_XEB