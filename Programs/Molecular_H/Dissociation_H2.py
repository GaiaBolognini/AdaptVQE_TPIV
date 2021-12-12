#Dissociation profile, Energy vs distance of the molecule H2

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt


#defintion of the molecule and of the fock initial state
symbols = ["H", "H"]
electrons, qubits = 2, 4
hf = qml.qchem.hf_state(electrons, qubits)
dev = qml.device("default.qubit", wires=qubits)

#defition of the circuit
def circuit(param, wires):
    qml.BasisState(hf, wires=wires)
    qml.DoubleExcitation(param, wires=[0,1,2,3])


#important: keyword arguments are never differntiated. When using the gradient descent included in pennylane
#all the numerical parameters appering in non-keyword arguments will be updated, while all numerical values included as keyword will not
@qml.qnode(dev)
def cost_fn(param, H = None):
    circuit(param, range(qubits))
    return qml.expval(H)

#classical optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.4)

# atomic distance
distance = np.linspace(0.5, 5, num=30)
Energy_best = []#storing correct values

#values for convergence in optimization
repetitions = 100
convergence = 10**(-7)

#parameters initialization
for i in range (len(distance)):
    theta = 0.0
    coordinates = np.array([0.0, 0.0, -distance[i] / 2, 0.0, 0.0, distance[i] / 2])
    Hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)

    energy_0 = cost_fn(theta, H=Hamiltonian)

    for k in range(repetitions):

        theta = opt.step(cost_fn, theta, H = Hamiltonian)

        energy_new = cost_fn(theta, H = Hamiltonian)

        # reached the convergence
        if (abs(energy_new - energy_0) < convergence):
            break

        energy_0 = energy_new

    Energy_best.append(energy_new)


print('Energy at minimum: ', min (Energy_best))

plt.figure()
plt.title("Dissociation profile")
plt.plot(distance, Energy_best, color = 'b', ls = '-', marker='o', markerfacecolor='red', label = 'predicted')
plt.xlabel(r"Distance, $\AA$")
plt.ylabel("Energy, Ha")
plt.legend()
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Dissociation_profile.pdf')
plt.show()











