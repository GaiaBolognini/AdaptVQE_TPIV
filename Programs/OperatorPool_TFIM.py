import pennylane as qml
from pennylane import numpy as np
import itertools

#definition of the TFIM Hamiltonian, with PBC
def TFIM_Hamiltonian(coeffs_ZZ, coeffs_X):
    """
    Definition of the Tranverse Field Hamiltonian: sum(Z_iZ_j) + sum(X_i)
    Args:
        coeffs_ZZ: coefficients in front of the ZZ operators, array
        coeffs_X: coefficients in front of the X operators, array

    Returns: the hamiltonian as a matrix

    """

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

def observables (qubits, observables_pool):
    """
    Observables of the operator pool, created in order to compute the commutator
    Args:
        qubits: number of qubits
        observables_pool: set of all possible observables ([])

    Returns: observable pool, updated with all the possible observables

    """
    #definition of the operators(Z, X, identity)
    Z = np.array([[1, 0], [0, -1]])
    X = np.array([[0, 1], [1, 0]])
    one = np.eye(2)

    for i in range(qubits):
        op1 = [Z] * ((i + 1) % qubits == 0) + [one] * (min(i, qubits - 2)) + [Z] + [Z] * ((i + 1) % qubits != 0) + [
            one] * (qubits - i - 2)
        M = 1
        for O in op1:
            M = np.kron(M, O)
        observables_pool.append(M)

        op2 = [one]*i + [X] + [one]*(qubits-i-1)
        M = 1
        for O in op2:
            M = np.kron(M, O)
        observables_pool.append(M)

    return observables_pool

#DEFINITION OF THE OPERATOR POOL (H):
# defined as the exponentialization of all the possible ZZ and X operators
#defined in a general way for n qubits

def ZZ_Operator(param, ij):
    """
    exp(-i theta/2 ZiZj)
    """
    qml.CNOT(wires=[ij[0], ij[1]])
    qml.RZ(param/2, wires = ij[1])
    qml.CNOT(wires=[ij[0], ij[1]])

def XX_Operator(param, ij):
    """
    exp(-i theta/2 XiXj), ij nearest neighbours
    """
    qml.Hadamard(wires=ij[0])
    qml.Hadamard(wires=ij[1])
    qml.CNOT(wires=[ij[0], ij[1]])
    qml.RZ(param/2, wires = ij[1])
    qml.CNOT(wires=[ij[0], ij[1]])
    qml.Hadamard(wires=ij[0])
    qml.Hadamard(wires=ij[1])

def ZY_Operator(param, ij):
    """
    exp(-i theta/2 ZiZj), ij nearest neighbours
    """
    qml.RX(-np.pi/2, wires=ij[1])
    qml.CNOT(wires=[ij[0], ij[1]])
    qml.RZ(param/2, wires = ij[1])
    qml.CNOT(wires=[ij[0], ij[1]])
    qml.RX(np.pi / 2, wires=ij[1])

def ZZX_Operator(param, ijk):
    """
    exp(-i theta/2 ZiZjXl), where i,j are nearest neighbours and l can be any other
    ijk=wires, ij=ZZ k=X.
    As ijk commutes it is possible to decide where X rotation applies (last wire)
    """
    qml.Hadamard(wires=ijk[-1])
    qml.CNOT(wires=[ijk[0], ijk[1]])
    qml.CNOT(wires=[ijk[1], ijk[2]])
    qml.RZ(param/2, wires = ijk[-1])
    qml.CNOT(wires=[ijk[1], ijk[2]])
    qml.CNOT(wires=[ijk[0], ijk[1]])
    qml.Hadamard(wires=ijk[-1])


def X_Operator(param, i):
    """
    exp(-i theta/2 Xi)
    """
    qml.RX(param/2, wires=i)

def ZZZZ_Operator(param, ijkl):
    """
    exp(-i theta ZiZjZkZl)
    """
    qml.broadcast(qml.CNOT, ijkl, pattern='chain')
    qml.RZ(param/2, wires=ijkl[-1])
    qml.CNOT(wires=[ijkl[2], ijkl[3]])
    qml.CNOT(wires=[ijkl[1], ijkl[2]])
    qml.CNOT(wires=[ijkl[0], ijkl[1]])


def making_operator_pool(qubits, operator_pool):
    """
    Given a certain number of qubits, defining the operator pool starting from X_operator and ZZ_oeprator
    Args:
        qubits: number of qubits
        operator_pool: set of all the operators

    Returns: operator_pool, a vector of functions with all the possible excitations
    """
    for i in range(qubits):
        wires = [i, i+1]
        if (i == qubits-1):
            wires=[i, 0]
        operator_ZZ = lambda param, wires = wires: ZZ_Operator(param, wires)
        operator_X = lambda param, wire=i: X_Operator(param, wire)
        operator_pool.append(operator_ZZ)
        operator_pool.append(operator_X)
    return operator_pool


#IMPLEMENTING H^2 OPERATOR POOL, defining all the possible operators.
def Operator_XX(qubits, operator_pool):
    """
    All the possible exp(-i theta XiXj) operators. ij can be any kind of wires.
    """
    combinations = itertools.combinations(range(qubits), 2)
    combinations = [i for i in combinations]
    for i in range (0, len(combinations), 1):
        operator = lambda param, wires = combinations[i]: XX_Operator(param, wires)
        operator_pool.append(operator)

def Operator_ZZ_notNN(qubits, operator_pool):
    """
    exp(-i theta ZiZj) operators, where ij are not nearest neighbours
    """
    combinations = itertools.combinations(range(qubits), 2)
    combinations = [i for i in combinations]
    index = []
    #eliminating the nearest neighnours couples, already implemented in the case of H
    for i in range(len(combinations)):
        if ((combinations[i][1] == combinations[i][0]+1) or (combinations[i][0] == 0 and combinations[i][1] == qubits-1)):
            index.append(i)
    combinations = np.delete(combinations, index, axis=0)
    combinations = combinations.tolist()
    for i in range (len(combinations)):
        operator= lambda param, wires = combinations[i]: ZZ_Operator(param, wires)
        operator_pool.append(operator)

def Operator_ZY(qubits, operator_pool):
    """
    exp(-i theta ZiYj) operators, where ij are nearest neighbours.
    The position of ZY (YZ) can be defined arbitrary the two operators commute
    """
    for i in range(qubits):
        wires = [i, i+1]
        if (i == qubits-1):
            wires=[i, 0]
        operator = lambda param, wires = wires: ZY_Operator(param, wires)
        operator_pool.append(operator)


def Operator_ZZX(qubits, operator_pool):
    """
    exp(-itheta (ZiZjZl)) where ij are nearest neighbours
    l is any other wire that is different from ij
    """
    for i in range(qubits):
        wires = [i, i+1]
        if (i == qubits-1):
            wires=[i, 0]
        #taking all the wires that are different from the already defined ones
        different_values = np.where((np.arange(qubits)!= wires[0]) & (np.arange(qubits)!= wires[1]))[0]
        for j in range (len(different_values)):
            operator = lambda param, wires = wires + [different_values[j]]: ZZX_Operator(param, wires)
            operator_pool.append(operator)



def Operator_ZZZZ(qubits, operator_pool):
    """
    exp(-i theta ZiZjZlZm) operators, where ijlm are all the different values
    param = parameter
    """
    combinations = itertools.combinations(range(qubits), 4)
    combinations = [i for i in combinations]
    for i in range(len(combinations)):
        operator = lambda param, wires = combinations[i]: ZZZZ_Operator(param, wires)
        operator_pool.append(operator)


