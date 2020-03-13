import numpy as np


# def AutoRotate(complex):
    # complex_out = complex
    # disp_complex = complex_out - complex_out[0]
    # for i in range(3):
        # max_disp_ind = np.argmax(np.abs(disp_complex))
        # disp_complex = complex_out - complex_out[max_disp_ind]
    # max_disp_ind = np.argmax(np.abs(disp_complex))
    # max_disp = disp_complex[max_disp_ind]
    # normalized_max_disp = max_disp / np.abs(max_disp)
    # # print(normalized_max_disp)
    # if normalized_max_disp.real < 0:
        # normalized_max_disp = -normalized_max_disp
    # CM = np.mean(complex_out) * 0  # origin
    # complex_out -= CM
    # complex_out /= normalized_max_disp
    # complex_out += CM
    # return complex_out


def complex_to_IQ(data_set):
    return np.transpose([np.real(data_set), np.imag(data_set)])
    
def IQ_to_complex(data_set):
    return np.squeeze(data_set[:,0] + 1j * data_set[:,1])
    
def find_angle(data_set):
    covariance_matrix = np.cov(np.real(data_set),np.imag(data_set))
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    ìndex = list(eigenvalues).index(min(eigenvalues))
    correct_eigenvector = eigenvectors[ìndex]
    return np.angle(correct_eigenvector[0] + 1j* correct_eigenvector[1])
    
def rotate(data_set):
    theta = find_angle(data_set)
    return data_set * np.exp(1j*theta)
    
def center(dataset):
    return dataset - np.mean(dataset)
    
def mapping(function, dataset, order=0):
    if order>0:
        return [mapping(function, d, order-1) for d in dataset]
    else:
        return np.array(list(map(function, dataset)))