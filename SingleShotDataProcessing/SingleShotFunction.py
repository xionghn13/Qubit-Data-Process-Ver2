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
    return params, params_err


def fit_blob_height(complex_array, centers, sigmas, bin=100, num_blob=4):
    sReal = np.real(complex_array)
    sImag = np.imag(complex_array)
    X, Y, H = complex_array_to_2d_histogram(complex_array, bin=bin)
    heights = get_center_heights(X, Y, H, centers)
    fixed_param = np.concatenate((centers, sigmas), axis=1)
    params, params_err = fg.fit_gaussian(X, Y, H, heights, method='fix_positions_and_widths', fixed_parameters=fixed_param)
    return params, params_err



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