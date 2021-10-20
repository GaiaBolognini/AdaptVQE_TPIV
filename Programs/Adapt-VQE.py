#implementing Adapt-VQE algorithm

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt

#definition of the moelcule and Hamiltonian
symbols = ["H", "H"]
distance = 1.3228
coordinates = np.array([0.0, 0.0, -distance/2, 0.0, 0.0, distance/2])
a_electrons = 2
a_orbitals = 2

Hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

singles, doubles = qchem.excitations(a_electrons, qubits)
excitation_operators = singles + doubles
print('Excitations: ', excitation_operators)
print('Total number of possible excitations: ', len(singles) + len(doubles))


#global variables definition
hf_state = qchem.hf_state(a_electrons, qubits)
dev = qml.device("default.qubit", wires = qubits)
dev_draw = qml.device('qiskit.aer', wires=qubits)
opt = qml.GradientDescentOptimizer(stepsize=0.5)

params_selected = []#parameters selected in the VQE agorithm
arg_selected = []#operators from the operator pool selected (which argument)
threshold = 10**(-4)
theta = 0.
s= 1/2
shift = np.pi/(4*s)
energy, phi = [], []

@qml.qnode(dev)
def circuit(param, wires, excitation=0):
    qml.BasisState(hf_state, wires=wires)

    for i in range(len(arg_selected)):
       if len(excitation_operators[arg_selected[i]]) == 4:
           qml.DoubleExcitation(params_selected[i], wires= excitation_operators[arg_selected[i]])
       else:
           qml.SingleExcitation(params_selected[i], wires=excitation_operators[arg_selected[i]])

    #adding the last operator
    if (len(excitation) == 4):
        qml.DoubleExcitation(param, wires=excitation)
    else:
        qml.SingleExcitation(param, wires=excitation)

    return qml.expval(Hamiltonian)



@qml.qnode(dev)
def circuit_VQE(params, wires=range(qubits)):
    qml.BasisState(hf_state, wires=wires)

    for i in range(len(arg_selected)):
        if len(excitation_operators[arg_selected[i]]) == 4:
            qml.DoubleExcitation(params[i], wires=excitation_operators[arg_selected[i]])
        else:
            qml.SingleExcitation(params[i], wires=excitation_operators[arg_selected[i]])

    return qml.expval(Hamiltonian)


for i in range(30):
    print(i)

    #evaluating the gradient changing the operator in the pool
    circuit_gradients = []
    for i in range(len(excitation_operators)):

        #evaluating the gradient using the grad function
        #ansatz = lambda param : circuit(param, wires=range(qubits), number=i)
        #grad = qml.grad(ansatz)
        #circuit_gradients.append(grad(theta))

        # evaluating the gradient using the parameter shift rule
        grad = s * (circuit(theta + shift, wires=range(qubits), excitation=excitation_operators[i]) - circuit(theta - shift, wires=range(qubits), excitation=excitation_operators[i]))
        # print(grad)
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
    convergence = 10 ** (-8)
    phi.append(theta)#parameter initialization

    energy_old = circuit_VQE(phi)
    print('Initial energy: ', energy_old)

    phi = opt.step(circuit_VQE, phi)
    energy_new = circuit_VQE(phi)

    while (abs(energy_new - energy_old) > convergence):
        energy_old = energy_new
        phi = opt.step(circuit_VQE, phi)
        energy_new = circuit_VQE(phi)

    print('Best parameter', phi)
    print('Optimized energy: ', energy_new)
    params_selected = phi
    energy.append(energy_new)



#building the final circuit and evaluating the final energy
@qml.qnode(dev)
def circuit_final(wires=range(qubits)):
    qml.BasisState(hf_state, wires=wires)
    # adding the chosen operators
    for i in range(len(arg_selected)):
       if len(excitation_operators[arg_selected[i]]) == 4:
           qml.DoubleExcitation(params_selected[i], wires=excitation_operators[arg_selected[i]])
       else:
           qml.SingleExcitation(params_selected[i], wires=excitation_operators[arg_selected[i]])

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
plt.ylabel("Energy, Ha")
plt.legend()
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Energy_Adapt_VQE.pdf')
plt.show()










