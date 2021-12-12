#implementing Adapt-VQE algorithm to find LiH ground state

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt


#definition of the moelcule and Hamiltonian
symbols = ["Li", "H"]
distance = 2.96928
coordinates = np.array([0.0, 0.0, 0.0, 0.0, 0.0, distance])
a_electrons = 2
a_orbitals = 5

Hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, active_electrons=a_electrons, active_orbitals=a_orbitals)
hf_state= qchem.hf_state(a_electrons, qubits)

#defining all the possible excitations:
singles, doubles = qchem.excitations(a_electrons, qubits)
excitation_operators = singles + doubles
#print(excitation_operators)
print('Total number of excitations: ', len(excitation_operators))

#global variables definition
dev = qml.device("default.qubit", wires = qubits)
dev_draw = qml.device('qiskit.aer', wires=qubits)
arg_selected = [] #storing the selected excitations
params_selected = []#parameters selected in the VQE agorithm

opt = qml.GradientDescentOptimizer(stepsize=0.5)
theta = 0.
threshold = 10**(-4)
convergence = 10 ** (-6)
s = 1/2
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


for i in range(20):
    print(i)

    #evaluating the gradient changing the excitation operator
    circuit_gradients = []
    for i in range(len(excitation_operators)):

        #evaluating the gradient using the parameter shift rule
        grad = s*(circuit(theta + shift, wires=range(qubits), excitation=excitation_operators[i]) - circuit(theta-shift, wires=range(qubits), excitation=excitation_operators[i]))
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
def circuit_final(wires=range(qubits)):
    qml.BasisState(hf_state, wires=wires)
    # adding the chosen operators
    for i in range(len(arg_selected)):
       if len(excitation_operators[arg_selected[i]]) == 4:
           qml.DoubleExcitation(params_selected[i], wires=excitation_operators[arg_selected[i]])
       else:
           qml.SingleExcitation(params_selected[i], wires=excitation_operators[arg_selected[i]])

    return qml.expval(Hamiltonian)

qnode_final = qml.QNode(circuit_final, dev)
drawer = qml.draw(qnode_final)
print('Circuit: ')
print(drawer())

qnode_final_draw = qml.QNode(circuit_final, dev_draw)
result = qnode_final_draw()
dev_draw._circuit.draw(output='mpl',filename = '/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Circuit_LiH_Adapt.pdf' )


# Saving the operators used and the correspondent parameters in a file
excitation_selected = []
for i in arg_selected:
    excitation_selected.append(excitation_operators[i])
np.savez('LiH_operators', excitation_selected, params_selected)

energy_final = qnode_final()
print('Ground state energy = ', energy_final)
print('How many operators: ', len(arg_selected))
print('Used operators: ', arg_selected)

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(12,8))
plt.title("Energy Optimization, LiH")
plt.plot(energy, 'o-')
plt.axhline(y=-7.882538, color='r', linestyle='-', label='Exact ground state energy')
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")
plt.legend()
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/LiH_adapt_VQE.pdf')
plt.show()


