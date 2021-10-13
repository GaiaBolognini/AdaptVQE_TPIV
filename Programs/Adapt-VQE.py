#implementing Adapt-VQE algorithm

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt
import Operator_Pool as OP

#definition of the moelcule and Hamiltonian
symbols = ["H", "H"]
distance = 1.3228
coordinates = np.array([0.0, 0.0, -distance/2, 0.0, 0.0, distance/2])
a_electrons = 2
a_orbitals = 2

Hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, active_electrons = a_electrons, active_orbitals = a_orbitals)

#global variables definition
hf_state = qchem.hf_state(a_electrons, qubits)
dev = qml.device("default.qubit", wires = qubits)
opt = qml.GradientDescentOptimizer(stepsize=0.02)

threshold = 10**(-2)
params_selected = []#parameters selected in the VQE agorithm
arg_selected = []#operators from the operator pool selected (which argument)
theta = -np.pi/4 #parameter chosen to evaluate the gradient
shift = 0.05
energy, phi = [], []

#operator pool
def apply (params, operator):
    return operator(params)

operator_pool = [OP.operator_1, OP.operator_2, OP.operator_3, OP.operator_4, OP.operator_5]

for i in range (60):
    print(i)

    #circuit definition to choose the new operator, returning the expectation value of the Hamiltonian
    @qml.qnode(dev)
    def circuit(param, wires, number=0):
        qml.BasisState(hf_state, wires=wires)
        #adding the already selected operators
        for j in range (len(arg_selected)):
            apply(params_selected[j], operator_pool[arg_selected[j]])

        #adding a different operator
        apply(param, operator_pool[number])

        return qml.expval(Hamiltonian)



    #evaluating the gradient chianging the operator in the pool
    circuit_gradients = []
    for i in range(len(operator_pool)):

        #evaluating the gradient using the grad function
        #ansatz = lambda param : circuit(param, wires=range(qubits), number=i)
        #grad = qml.grad(ansatz)
        #circuit_gradients.append(grad(theta))

        #evaluating the gradient using the parameter shift rule
        grad = circuit(theta+shift, wires=range(qubits),number=i) - circuit(theta-shift, wires=range(qubits) , number=i)
        circuit_gradients.append(grad)

    #print('Gradient: ', circuit_gradients)

    #taking tha max value of the gradient
    max_grad = max(np.abs(circuit_gradients))

    #evaluating with respect to the threshold
    if (max_grad < threshold):
        break

    #saving the argument selected in arg_selected
    arg_selected.append(np.argmax(np.abs(circuit_gradients)))



    #applying the VQE to find the best set of parameters:
    @qml.qnode(dev)
    def circuit_VQE(params, wires=range(qubits)):
        qml.BasisState(hf_state, wires=wires)
        # adding the chosen operators
        for j in range(len(arg_selected)):
            apply(params[j], operator_pool[arg_selected[j]])

        return qml.expval(Hamiltonian)


    convergence = 10 ** (-3)
    phi.append(-np.pi/4)#parameter initialization

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
@qml.qnode(dev)
def circuit_final(wires=range(qubits)):
    qml.BasisState(hf_state, wires=wires)
    # adding the chosen operators
    for j in range(len(arg_selected)):
        apply(params_selected[j], operator_pool[arg_selected[j]])

    return qml.expval(Hamiltonian)


drawer = qml.draw(circuit_final)
print('Circuit: ')
print(drawer())

energy_final = circuit_final()
print('Ground state energy = ', energy_final)
print('How many operators: ', len(arg_selected))
print('Used operators: ', arg_selected)

plt.figure()
plt.title("Energy Optimization")
plt.plot(energy, 'o-')
plt.xlabel("Iterations")
plt.ylabel("Energy, a.u.")
plt.legend()
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Energy_Adapt_VQE.pdf')
plt.show()










