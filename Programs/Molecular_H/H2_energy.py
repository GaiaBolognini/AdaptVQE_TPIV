#algorithm using VQE to obtain the dissociation profìle of the hydrogen molecule

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt


#we first specify the molecule we want to simulate: H2 and its coordinates
symbols = ["H", "H"]
distance = 1.3228
coordinates = np.array([0.0, 0.0, -distance/2, 0.0, 0.0, distance/2])

Hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

print('Number of qubits: ', qubits)
print("Hamiltonian ", Hamiltonian)

#implementing the VQE algorithm
dev = qml.device("default.qubit", wires=qubits)
dev_draw = qml.device('qiskit.aer', wires=qubits)


#Hartree-Fock state as initial state
electrons = 2
hf = qml.qchem.hf_state(electrons, qubits)
print('Fock state: ', hf)


def circuit(param, wires):
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(param, wires=[0,1,2,3])

#definition of the cost function
def cost_fn (param):
    circuit(param, range(qubits))
    return qml.expval(Hamiltonian)

qnode = qml.QNode(cost_fn, dev)

#create a function that draws the given qnode
drawer = qml.draw(qnode)
print('Circuit: ')
print(drawer(param=np.pi))

#printing in matplotlib style
qnode_draw = qml.QNode(cost_fn, dev_draw)
result = qnode_draw(0)
dev_draw._circuit.draw(output='mpl',filename = '/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Circuit_H2.pdf' )

#definition of a classical optimizer, offered by PL
# x(t+1) = x(t) - n*grad(function(x(t)), where n is the size of our step
opt = qml.GradientDescentOptimizer(stepsize=0.4)

#inizialing the parameter
theta = 0.0
params = [theta]

energy = [qnode(theta)]
print("Energy: ",energy)

repetitions = 100
convergence = 10**(-7)

for i in range (repetitions):

    theta = opt.step(qnode, theta)

    energy_new = qnode(theta)
    energy.append(energy_new)
    params.append(theta)

    #reached the convergence
    if (abs(energy_new - energy[i]) < convergence):
        break

print('Best Energy value: %.4f Hartree' %(energy[-1]))
print('Best parameter value %.4f: '%(params[-1]))

fig = plt.figure(figsize = (10,4))
ax_1 = fig.add_subplot(1,2,1)
plt.title("Energy optimization")
plt.plot(energy, 'o-')
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")

ax_2 = fig.add_subplot(1,2,2)
plt.title("Parameters optimization")
plt.plot(params, 'o-')
plt.xlabel("Iterations")
plt.ylabel(r"Parameter $\theta$")

plt.savefig("/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/H2_optimization.pdf")
plt.show()








