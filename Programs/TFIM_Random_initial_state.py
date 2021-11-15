#implementing TFIM Hamiltonian and finding its ground state:
#initializing with a random rotation where the angles are different for each qubit


import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt
import OperatorPool_TFIM as OP

qubits = 4
dev = qml.device("default.qubit", wires = qubits)
dev_draw = qml.device('qiskit.aer', wires=qubits)
opt = qml.GradientDescentOptimizer(stepsize=0.5)

threshold = 5*10**(-3) #for the gradient
convergence=10**(-5) #for the VQE algorithm
params_selected = []#values of the parameters selected in the VQE agorithm
arg_selected = []#operators from the operator pool selected (which argument)
theta = 0.0 #initialization of the parameter used in the VQE algorithm
energy, phi = [], []

#defining the Hamiltonian of the problem
X_coefs = [1.0]*qubits
ZZ_coefs = [1.0]*qubits
TFIM_H = OP.TFIM_Hamiltonian(ZZ_coefs, X_coefs)
print('TFIM Hamiltonian: \n', TFIM_H)

#finding the eigenvalues of the Hamiltonian:
eigenvals,eigenvects=np.linalg.eigh(TFIM_H)
print('Eigeinvalues: ', eigenvals)
ground_state = min(np.real(eigenvals))
print('Ground state energy: ', ground_state)


#Definition of the operator pool
def apply(params, operator):
    return operator(params)

operator_pool = []

#creating H operator pool
operator_pool = OP.making_operator_pool(qubits, operator_pool)

#creating H^2 operator pool
OP.Operator_XX(qubits, operator_pool)
OP.Operator_ZZ_notNN(qubits, operator_pool)
OP.Operator_ZY(qubits, operator_pool)
OP.Operator_ZZX(qubits, operator_pool)
OP.Operator_ZZZZ(qubits, operator_pool)

#new operator defintion (not belonging to the Hamiltonian)
OP.Operator_XY(qubits, operator_pool)
OP.Operator_ZX(qubits, operator_pool)
OP.Operator_YY(qubits, operator_pool)
OP.Operator_ZXY(qubits, operator_pool)
OP.Operator_ZXY_permutations(qubits, operator_pool)

print('Dimension of the operator pool', len(operator_pool))

#DEFINITION OF THE INITIAL STATE
params_initial = np.pi * np.random.rand(qubits, 3)

@qml.qnode(dev)
def circuit(param, wires=range(qubits), number=0):
    """
    Circuit definition to choose the new operator
    Args:
        param: parameter to be passed
        wires:
        number: corresponding element in the operator pool to apply

    Returns:expectation value of the Hamiltonian

    """
    for i in range (qubits):
        angles = params_initial[i][:]
        qml.Rot(angles[0], angles[1], angles[2], wires=i)

    # adding the already selected operators
    for j in range(len(arg_selected)):
        apply(params_selected[j], operator_pool[arg_selected[j]])

    # adding a different operator
    apply(param, operator_pool[number])
    return qml.expval(qml.Hermitian(TFIM_H, wires=range(qubits)))

drawer = qml.draw(circuit)
print('Circuit: ')
print('Result: ', circuit(param=0, number=10))
print(drawer(param=0, number=10))


@qml.qnode(dev)
def circuit_VQE(params, wires=range(qubits)):
    """
    Circuit definition to compute the VQE algorithm
    Returns: Expectation value of the Hamiltonian
    """
    for i in range (qubits):
        angles = params_initial[i][:]
        qml.Rot(angles[0], angles[1], angles[2], wires=i)

    #adding the chosen operators
    for j in range(len(arg_selected)):
        apply(params[j], operator_pool[arg_selected[j]])

    return qml.expval(qml.Hermitian(TFIM_H, wires=range(qubits)))

#running the VQE-Adapt algorithm
for i in range(30):
    print(i)
    circuit_gradients = []

    #evaluating the gradient using the grad function
    for i in range(len(operator_pool)):
        ansatz = lambda param, number=i: circuit(param, wires=range(qubits), number=number)
        grad = qml.grad(ansatz)
        circuit_gradients.append(grad(theta))

    print('Gradient: ', circuit_gradients)

    #taking tha max value of the gradient
    max_grad = max(np.abs(circuit_gradients))

    #evaluating with respect to the threshold
    if (max_grad < threshold):
        break

    #saving the argument selected in arg_selected
    arg_selected.append(np.argmax(np.abs(circuit_gradients)))
    print(arg_selected)

    #applying the VQE to find the best set of parameters
    phi.append(theta)#parameter initialization

    energy_old = circuit_VQE(phi)
    print('Initial energy: ', energy_old)

    phi = opt.step(circuit_VQE, phi)
    energy_new = circuit_VQE(phi)

    while (abs(energy_new - energy_old) > convergence):
        energy_old = energy_new
        phi = opt.step(circuit_VQE, phi)
        energy_new = circuit_VQE(phi)

    #print('Best parameter', phi)
    print('Optimized energy: ', energy_new)
    params_selected = phi
    energy.append(energy_new)


#building the final circuit and evaluating the final energy
def circuit_final(wires=range(qubits)):
    """
    Final circuit: built with the selected operators and parameters
    Returns:Expectation value of the Hamiltonian
    """
    for i in range (qubits):
        angles = params_initial[i][:]
        qml.Rot(angles[0], angles[1], angles[2], wires=i)
    # adding chosen operators and parameters
    for j in range(len(arg_selected)):
        apply(params_selected[j], operator_pool[arg_selected[j]])

    return qml.expval(qml.Hermitian(TFIM_H, wires=range(qubits)))

qnode_final = qml.QNode(circuit_final, dev)
drawer = qml.draw(qnode_final)
print('Circuit: ')
print(drawer())

qnode_final_draw = qml.QNode(circuit_final, dev_draw)
result = qnode_final_draw()
dev_draw._circuit.draw(output='mpl',filename = '/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Circuit_TFIM_random_intial_state.pdf' )


@qml.qnode(dev)
def circuit_final_state(wires=range(qubits)):
    """
    Circuit to check the final state
    Returns: probabilities for all the combination of possible results
    """
    for i in range (qubits):
        angles = params_initial[i][:]
        qml.Rot(angles[0], angles[1], angles[2], wires=i)

    # adding the chosen operators
    for j in range(len(arg_selected)):
        apply(params_selected[j], operator_pool[arg_selected[j]])

    return qml.probs(wires=range(qubits))

print('State probabilities: ', circuit_final_state())

energy_final = qnode_final()
print('Ground state energy = ', energy_final)
print('How many operators: ', len(arg_selected))
print('Used operators: ', arg_selected)
print('Parameters: ', params_selected)

plt.figure()
plt.title("Energy Optimization")
plt.axhline(y=ground_state, color='r', linestyle='-', label='Ground state energy')
plt.plot(energy,'o-')
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")
plt.legend()
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/TFIM_optimization_random_initial_state.pdf')
plt.show()
