import pennylane as qml
from pennylane import numpy as np

#definition of the TFIM Hamiltonian, with PBC
def TFIM_Hamiltonian(coeffs_ZZ, coeffs_X):
    #definition of the operators(Z, X, identity)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    one = np.eye(2)

    n_qubit = len(coeffs_X)
    dim = 2 ** n_qubit

    #Hamiltonian
    H = np.zeros((dim, dim))

    #builds the Hamiltonian just adding all the possible terms in the previous sum
    for i in range(len(coeffs_X)):
        op1 = [Z] * ((i + 1) % n_qubit == 0) + [one] * (min(i, n_qubit - 2)) + [Z] + [Z] * ((i + 1) % n_qubit != 0) + [
            one] * (n_qubit - i - 2)
        op2 = [one] * i + [X] + [one] * (n_qubit - i - 1)
        M = 1
        for O in op1:
            M = np.kron(M, O)
        H += M * coeffs_ZZ[i]

        M = 1
        for O in op2:
            M = np.kron(M, O)
        H += M * coeffs_X[i]

    return H

#definiton of the operator pool: defined as the exponentialization of all the possible ZZ and X operators
#trying to define it to be applied to a generic number of qubits n
def ZZ_Operator(params, wires):
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(params, wires = wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def X_Operator(params, wire):
    qml.RX(params, wires=wire)

def making_operator_pool(qubits, operator_pool):
    for i in range(qubits):
        wires=[i, i+1]
        if (i==qubits-1):
            wires=[i, 0]
        operator_ZZ = lambda param, wires=wires: ZZ_Operator(param, wires)
        operator_X = lambda param, wire=i: X_Operator(param, wire)
        operator_pool.append(operator_ZZ)
        operator_pool.append(operator_X)
    return operator_pool

#operator ZiZjZiZj
def Operator_0 (param, wires):
    qml.SWAP(wires=wires)
    qml.RZ(param, wires=wires[1])
    qml.SWAP(wires=wires)

#operator ZiZjZjZl, defining the wires in the form [[0,1][1,2]]
def Operator_1 (param, wires):
    qml.CNOT(wires=wires[0])
    qml.CNOT(wires=wires[1])
    qml.RZ(param, wires=wires[1][1])
    qml.CNOT(wires=wires[1])
    qml.CNOT(wires=wires[0])

#operator ZiZjZlZm
def Operator_2 (param, wires):
    qml.CNOT(wires=wires[0])
    qml.CNOT(wires=[wires[0][1], wires[1][0]])
    qml.CNOT(wires=wires[1])
    qml.RZ(param, wires=wires[1][1])
    qml.CNOT(wires=wires[1])
    qml.CNOT(wires=[wires[0][1], wires[1][0]])
    qml.CNOT(wires=wires[0])

#operator ZiZjXl, wires in the form [i, j] and then wire=l
def Operator_3(param, wires, wire):
    qml.CNOT(wires=wires)
    qml.Hadamard(wires=wire)
    qml.CNOT(wires=[wires[1], wire])
    qml.RZ(param, wires=wire)
    qml.CNOT(wires=[wires[1], wire])
    qml.Hadamard(wires=wire)
    qml.CNOT(wires=wires)
