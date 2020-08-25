#Generic Python packages
import numpy as np
import qutip
from scipy import optimize
import sys
import matplotlib.pyplot as plt

    

#Generic Quantum operations

def apply_unitary(density_matrix, *args):
    d = density_matrix
    for U in args:
        d = apply_single_unitary(d, U)
    return d

def apply_gate(density_matrix, *args):
    d = density_matrix
    for g in args:
        d = apply_single_gate(d, g)
    return d

def gate_to_unitary(gate):
    # there should be a minus sign here?
    # return (1j*gate/2).expm()
    return (-1j*gate/2).expm()


def apply_single_unitary(density_matrix,unitary):
    return unitary * density_matrix * unitary.dag()
    
def apply_single_gate(density_matrix,gate):
    unitary = gate_to_unitary(gate)
    return unitary * density_matrix * unitary.dag()
    
# Single Qubit stuff

Id = qutip.qeye(2)
Sx = qutip.sigmax()
Sy = qutip.sigmay()
Sz = qutip.sigmaz()

def single_qubit_density_matrix(x,y,z):
    return (Id + x * Sx + y * Sy + z * Sz)/2
    
# Two qubit functions
ListGateAllXY = ["I-I", "Xp-Xp", "Yp-Yp", "Xp-Yp", "Yp-Xp", "X2p-I", "Y2p-I", "X2p-Y2p", "Y2p-X2p",
                 "X2p-Yp", "Y2p-Xp", "Xp-Y2p", "Yp-X2p", "X2p-Xp", "Xp-X2p", "Y2p-Yp", "Yp-Y2p", "Xp-I",
                 "Yp-I", "X2p-X2p", "Y2p-Y2p"]

def decompose_in_basis(matrix, basis_list):
    n = len(basis_list)
    passage_matrix = np.zeros((n, n), dtype=complex)
    vector = np.zeros((n, 1), dtype=complex)
    for i in range(n):
        vector[i] = np.matrix(basis_list[i] * matrix).trace()
        for j in range(n):
            passage_matrix[i, j] = (basis_list[i] * basis_list[j]).tr()
#     print(passage_matrix)
    passage_matrix = np.linalg.inv(passage_matrix)
    return (passage_matrix.dot(vector)).ravel()

list_gates_tomo_single_qubit = ["I", "X2p", "Y2m"]   


list_gates_tomo_two_qubits_16 = ["I-I", "I-Xp", "I-Y2p", "I-X2m", 
    "Xp-I", "Xp-Xp", "Xp-Y2p", "Xp-X2m", "Y2p-I", "Y2p-Xp", "Y2p-Y2p",
     "Y2p-X2m", "X2m-I", "X2m-Xp", "X2m-Y2p", "X2m-X2m"]    

           
list_gates_tomo_two_qubits_36 = ["I-I", "Xp-I", "X2p-I", "X2m-I", "Y2p-I", 
   "Y2m-I", "I-Xp", "Xp-Xp", "X2p-Xp", "X2m-Xp", "Y2p-Xp", "Y2m-Xp", 
   "I-X2p", "Xp-X2p", "X2p-X2p", "X2m-X2p", "Y2p-Y2p", "Y2m-Y2p", 
   "I-X2m", "Xp-X2m", "X2p-X2m", "X2m-X2m", "Y2p-X2m", "Y2m-X2m", 
   "I-Y2p", "Xp-Y2p", "X2p-Y2p", "X2m-Y2p", "Y2p-Y2p", "Y2m-Y2p", 
   "I-Y2m", "Xp-Y2m", "X2p-Y2m", "X2m-Y2m", "Y2p-Y2m", "Y2m-Y2m"]

dictionary_single_qubit_gates = {'I': Id, 'X2p': Sx * np.pi / 2, 'Xp': Sx * np.pi, 'X2m': -Sx * np.pi / 2, 'Xm': -Sx * np.pi,
                                  'Y2p': Sy * np.pi / 2, 'Yp': Sy * np.pi, 'Y2m': -Sy * np.pi / 2, 'Ym': -Sy * np.pi,
                                  'Z2p': Sz * np.pi / 2, 'Zp': Sz * np.pi, 'Z2m': -Sz * np.pi / 2, 'Zm': -Sz * np.pi}

dictionary_two_qubit_gates = {}
dictionary_two_qubit_gates_unitary = {}
dictionary_two_qubit_gates_unitary_matrix = {}

for gate in list_gates_tomo_two_qubits_36:
    [string1, string2] = gate.split('-')
    two_qubit_gate = qutip.tensor(dictionary_single_qubit_gates[string1], Id) + \
                                        qutip.tensor(Id, dictionary_single_qubit_gates[string2])
    dictionary_two_qubit_gates[gate] = two_qubit_gate
    dictionary_two_qubit_gates_unitary[gate] = gate_to_unitary(two_qubit_gate)
    dictionary_two_qubit_gates_unitary_matrix[gate] = np.matrix(gate_to_unitary(two_qubit_gate))
# two single qubit gate at the same time


operator_basis_single_qubit = [1 / np.sqrt(2.0) * Id, 1 / np.sqrt(2.0) * Sx, 1 / np.sqrt(2.0) * Sy,
                               1 / np.sqrt(2.0) * Sz]

operator_basis_two_qubits = []
for P in operator_basis_single_qubit:
    for Q in operator_basis_single_qubit:
        operator_basis_two_qubits += [qutip.tensor(P, Q)]
        
        
default_initial_gates = []
default_state_basis = []
for gate_string in list_gates_tomo_two_qubits_16:
    gate = dictionary_two_qubit_gates[gate_string]
    default_initial_gates += [gate]    
    state = qutip.ket2dm(qutip.basis(2, 0))
    state = qutip.tensor(state, state)
    state = apply_gate(state, gate)
    default_state_basis += [state]
    
default_beta = np.zeros((len(operator_basis_two_qubits), len(operator_basis_two_qubits), len(default_state_basis),
                         len(default_state_basis)), dtype=complex)
for j in range(len(default_state_basis)):
    for n in range(len(operator_basis_two_qubits)):
        for m in range(len(operator_basis_two_qubits)):
            default_beta[m, n, j, :] = decompose_in_basis(operator_basis_two_qubits[m] * default_state_basis[j]\
                                                          * operator_basis_two_qubits[n].dag(), default_state_basis)

## tomography functions

def transmission_to_beta_coefficients(transmission_vector):
    zthA = -1
    zthB = -1
    matrix = np.array([
        [1, zthA, zthB, zthA * zthB],
        [1, -zthA, zthB, -zthA * zthB],
        [1, zthA, -zthB, -zthA * zthB],
        [1, -zthA, -zthB, zthA * zthB]
    ])
    # [gg, eg, ge, ee].T
    # I-I, Xp-I, I-Xp, Xp-Xp
    return (np.linalg.inv(matrix).dot(transmission_vector)).ravel()


def two_qubit_measurement_operator(beta_list):
    [beta_II, beta_ZI, beta_IZ, beta_ZZ] = beta_list
    return beta_II * qutip.tensor(Id, Id) + beta_ZI * qutip.tensor(Sz, Id) + beta_IZ * qutip.tensor(Id, Sz)\
            + beta_ZZ * qutip.tensor(Sz, Sz)



def theoretical_measurement_results(rho, beta_list, gate_list):
    M = np.matrix(two_qubit_measurement_operator(beta_list))
    result = np.zeros(len(gate_list), dtype=complex)
    for i in range(len(gate_list)):
#         transformed_density_matrix = np.matrix(apply_single_gate(rho, dictionary_two_qubit_gates[gate_list[i]]))
        unitary_matrix = dictionary_two_qubit_gates_unitary_matrix[gate_list[i]]
        transformed_density_matrix = unitary_matrix.dot(rho).dot(unitary_matrix.H)
        result[i] = (transformed_density_matrix.dot(M)).trace()[0, 0]
    return result


def calculate_chi_for_two_qubits_unitary(unitary):
    lambda_coefficients = []
    for op in operator_basis_two_qubits:
        lambda_coefficients += [0.5 * (unitary * op).tr()]
    lambda_coefficients = np.matrix(lambda_coefficients)
    return lambda_coefficients.H.dot(lambda_coefficients)


def find_most_likely_density_matrix(data, beta_coefficients, print_error=True, options={'maxiter': 100, 'gtol': 1e-03}):
    def construct_rho_from_T(param):
        [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14 ,t15, t16] = param        
        T = np.matrix([
            [t1, 0, 0, 0],
            [t5 + 1j * t6, t2, 0, 0],
            [t11 + 1j * t12, t7 + 1j * t8, t3, 0],
            [t15 + 1j * t16, t13 + 1j * t14, t9 + 1j * t10, t4]
        ])
        rho = T.H.dot(T)        
        return rho / rho.trace()
        
    def error_func(param):
        rho = construct_rho_from_T(param)
        data_theory = theoretical_measurement_results(rho, beta_coefficients, list_gates_tomo_two_qubits_36)
        err = np.sum(np.abs(data_theory - data) ** 2)
        
        if print_error:
            string = 'error=%.3G....' % err
            sys.stdout.write("\r")
            sys.stdout.write(string)
            sys.stdout.flush()

        return err
    p0 = np.zeros(16)
    p0[:4] = 1
#     res = optimize.minimize(error_func, p0, method='Powell',tol=1.e-4,
#             options={'maxiter': 100, 'maxfev': 1000, 'ftol': 1e-4})
    res = optimize.minimize(error_func, p0, method='CG',
                           options=options)
    p = res.x
    return construct_rho_from_T(p)

def calculate_chi_from_density_matrices(list_density_matrices, initial_gates=default_initial_gates):
    if initial_gates == default_initial_gates:
        state_basis = default_state_basis
        beta = default_beta
    else:
        state_basis = []
        for gate_string in initial_gates:
            gate = dictionary_two_qubit_gates[gate_string]
            state = qutip.ket2dm(qutip.basis(2, 0))
            state = qutip.tensor(state, state)
            state = apply_gate(state, gate)
            state_basis += [state]
    
        beta = np.zeros((len(operator_basis_two_qubits), len(operator_basis_two_qubits), len(state_basis),
                         len(state_basis)), dtype=complex)
        for j in range(len(state_basis)):
            for n in range(len(operator_basis_two_qubits)):
                for m in range(len(operator_basis_two_qubits)):
                    default_beta[m, n, j, :] = decompose_in_basis(operator_basis_two_qubits[m] * state_basis[j]\
                                                                  * operator_basis_two_qubits[n].dag(), state_basis)
    lambda_matrix = np.zeros((len(list_density_matrices), len(state_basis)), dtype=complex)
    for i in range(len(list_density_matrices)):
        lambda_matrix[i, :] = decompose_in_basis(list_density_matrices[i], state_basis)
    lambda_vector = lambda_matrix.reshape((len(list_density_matrices) * len(state_basis), 1))
    shape_beta = beta.shape
    beta_matrix = beta.reshape((shape_beta[0] * shape_beta[1], shape_beta[2] * shape_beta[3])).T
    chi_vector = (np.linalg.inv(beta_matrix)).dot(lambda_vector)
    return chi_vector.reshape((shape_beta[0], shape_beta[1]))

def show_density_matrix_2D(matrix):
    list_ticks_1QB = ['1', '0']
    list_ticks_2QB = []
    for t1 in list_ticks_1QB:
        for t2 in list_ticks_1QB:
            list_ticks_2QB += [t1 + t2]
    position_array = np.linspace(0, 3, 4)
    plt.figure()
    ax = plt.subplot(121)
    plt.imshow(matrix.real, cmap=plt.cm.bwr, vmin=-1, vmax=1)
    ax.xaxis.tick_top()
    plt.xticks(position_array, list_ticks_2QB)
    plt.yticks(position_array, list_ticks_2QB)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim((position_array.max() + 0.5, position_array.min() - 0.5))

    ax = plt.subplot(122)
    plt.imshow(matrix.imag, cmap=plt.cm.bwr, vmin=-1, vmax=1)
    ax.xaxis.tick_top()
    plt.xticks(position_array, list_ticks_2QB)
    plt.yticks(position_array, list_ticks_2QB)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim((position_array.max() + 0.5, position_array.min() - 0.5))

    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.1, 0.01, 0.8])
    plt.colorbar(cax=cax);
    
def show_chi_matrix_2D(matrix):
    list_ticks_1QB = ['I', 'X', 'Y', 'Z']
    list_ticks_2QB = []
    for t1 in list_ticks_1QB:
        for t2 in list_ticks_1QB:
            list_ticks_2QB += [t1 + t2]
    position_array = np.linspace(0, 15, 16)
    max_chi = np.abs(matrix).max()
    plt.figure()
    ax = plt.subplot(121)
    plt.imshow(matrix.real, cmap=plt.cm.bwr, vmin=-max_chi, vmax=max_chi)
    ax.xaxis.tick_top()
    plt.xticks(position_array, list_ticks_2QB)
    plt.yticks(position_array, list_ticks_2QB)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim((15.5, -0.5))

    ax = plt.subplot(122)
    plt.imshow(matrix.imag, cmap=plt.cm.bwr, vmin=-max_chi, vmax=max_chi)
    ax.xaxis.tick_top()
    plt.xticks(position_array, list_ticks_2QB)
    plt.yticks(position_array, list_ticks_2QB)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim((15.5, -0.5))

    plt.subplots_adjust(bottom=0.1, right=0.9, top=0.9)
    cax = plt.axes([0.95, 0.1, 0.01, 0.8])
    plt.colorbar(cax=cax);
    
def fidelity_chi(chi1_ndarray, chi2_ndarray, d=2 ** 2):
    chi1 = np.matrix(chi1_ndarray)
    chi2 = np.matrix(chi2_ndarray)
    chi1_renormalized = chi1 / np.sqrt((chi1.H.dot(chi1)).trace()) * d
    chi2_renormalized = chi2 / np.sqrt((chi2.H.dot(chi2)).trace()) * d
    fid = ((chi1_renormalized.H.dot(chi2_renormalized)).trace() / d + 1) / (d + 1)
    fid = fid.real
    return fid 


# DataAnalysis Functions
    
def AllXYMetric(dataset):
    if not(len(dataset)==21):
        raise ValueError("There should be 21 AllXY gates !")
    else:
        m1 = np.mean(dataset[0:5])
        m2 = np.mean(dataset[5:17])
        m3 = np.mean(dataset[17:21])
        minimum = min(dataset)
        maximum = max(dataset)
        
        return (np.linalg.norm(dataset[0:5]-m1) + np.linalg.norm(dataset[5:17]-m2) + np.linalg.norm(dataset[17:21]-m3))/(maximum-minimum)