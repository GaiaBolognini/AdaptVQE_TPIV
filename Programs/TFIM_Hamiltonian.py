#implementing TFIM Hamiltonian and finding its ground state

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt
import OperatorPool_TFIM as OP

qubits=4
hf_state= qchem.hf_state(2, qubits)
singles, doubles = qchem.excitations(2, qubits)
excitation_operators = singles + doubles

dev = qml.device("default.qubit", wires = qubits)
dev_draw = qml.device('qiskit.aer', wires=qubits)
opt = qml.GradientDescentOptimizer(stepsize=0.01)

threshold = 10**(-5)
params_selected = []#parameters selected in the VQE agorithm
arg_selected = []#operators from the operator pool selected (which argument)
s= 1/2
shift = np.pi/(4*s)
energy, phi = [], []
convergence=10**(-5)

#defining the Hamiltonian of the problem
X_coefs = [1.0]*qubits
ZZ_coefs = [1.0]*qubits
TFIM_H = OP.TFIM_Hamiltonian(ZZ_coefs, X_coefs)
print('TFIM Hamiltonian: ', TFIM_H)


#finding the eigenvalues of the Hamiltonian:
eigenvals,eigenvects=np.linalg.eigh(TFIM_H)
print('Eigeinvalues: ', eigenvals)
ground_state = min(np.real(eigenvals))
print('Ground state energy: ', ground_state)


#Definition of the operator pool
def apply (params, operator):
    return operator(params)

operator_pool=[]
operator_pool = OP.making_operator_pool(qubits, operator_pool)
operator_pool.append(lambda param, wires=[1,2]: OP.Operator_0(param, wires))
operator_pool.append(lambda param, wires=[[0,1], [1,2]]: OP.Operator_1(param, wires))
operator_pool.append(lambda param, wires=[[0,1],[2,3]]: OP.Operator_2(param, wires))
operator_pool.append(lambda param, wires=[0,1], wire=3: OP.Operator_3(param, wires, wire))
print(len(operator_pool))

# circuit definition to choose the new operator, returning the expectation value of the Hamiltonian
@qml.qnode(dev)
def circuit(param, wires=range(qubits), number=0):
    for i in range(int(qubits)):
        qml.Hadamard(wires=i)
    # adding the already selected operators
    for j in range(len(arg_selected)):
        apply(params_selected[j], operator_pool[arg_selected[j]])

    # adding a different operator
    apply(param, operator_pool[number])

    return qml.expval(qml.Hermitian(TFIM_H, wires=range(qubits)))

drawer = qml.draw(circuit)
print('Circuit: ')
print('Result: ', circuit(param=-np.pi/2, number=5))
print(drawer(param=-np.pi/4, number=5))


@qml.qnode(dev)
def circuit_VQE(params, wires=range(qubits)):
    for i in range(int(qubits)):
        qml.Hadamard(wires=i)
    # adding the chosen operators
    for j in range(len(arg_selected)):
        apply(params[j], operator_pool[arg_selected[j]])

    return qml.expval(qml.Hermitian(TFIM_H, wires=range(qubits)))


for i in range(40):
    print(i)
    if (i<10):
      theta=-np.pi/4
    else:
      theta=0.0
    #evaluating the gradient changing the operator in the pool
    circuit_gradients = []
    for i in range(len(operator_pool)):
        #evaluating the gradient using the parameter shift rule
        grad = s*(circuit(theta+shift, wires=range(qubits),number=i) - circuit(theta-shift, wires=range(qubits), number=i))
        circuit_gradients.append(grad)

    print('Gradient: ', circuit_gradients)

    #taking tha max value of the gradient
    max_grad = max(np.abs(circuit_gradients))

    #evaluating with respect to the threshold
    if (max_grad < threshold):
        break

    #saving the argument selected in arg_selected
    arg_selected.append(np.argmax(np.abs(circuit_gradients)))


    #applying the VQE to find the best set of parameters
    convergence = 10 ** (-4)
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
    for i in range(qubits):
        qml.Hadamard(wires=i)
    # adding the chosen operators
    for j in range(len(arg_selected)):
        apply(params_selected[j], operator_pool[arg_selected[j]])

    return qml.expval(qml.Hermitian(TFIM_H, wires=range(qubits)))

qnode_final = qml.QNode(circuit_final, dev)
drawer = qml.draw(qnode_final)
print('Circuit: ')
print(drawer())

qnode_final_draw = qml.QNode(circuit_final, dev_draw)
result = qnode_final_draw()
dev_draw._circuit.draw(output='mpl',filename = '/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Circuit_TFIM_OP_allargata.pdf' )

@qml.qnode(dev)
def circuit_final_state(wires=range(qubits)):
    for i in range(qubits):
        qml.Hadamard(wires=i)
    # adding the chosen operators
    for j in range(len(arg_selected)):
        apply(params_selected[j], operator_pool[arg_selected[j]])

    return qml.probs(wires=range(qubits))

print('Probability: ', circuit_final_state())

energy_final = qnode_final()
print('Ground state energy = ', energy_final)
print('How many operators: ', len(arg_selected))
print('Used operators: ', arg_selected)

plt.figure()
plt.title("Energy Optimization")
#energy_plot = energy[1:]
plt.plot(energy)
plt.axhline(y=ground_state, color='r', linestyle='-', label='Ground state energy')
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")
plt.legend()
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/TFIM_optimization_OP_allargata.pdf')
plt.show()
