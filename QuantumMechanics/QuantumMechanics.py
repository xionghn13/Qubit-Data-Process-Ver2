#Generic Python packages
import numpy as np
import qutip

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
    return (1j*gate/2).expm()

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