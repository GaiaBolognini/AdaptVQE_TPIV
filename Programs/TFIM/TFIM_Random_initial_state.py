#implementing TFIM Hamiltonian and finding its ground state through the Adapt-VQE algorithm
#initializing the state with a random rotation
#Possible to sample from the complete operatro pool
#taking one oeprator at a time to perform the algorithm
#using the gradient descent optimizer

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import OperatorPool_TFIM as OP
from random import sample
from random import seed
import pandas as pd

prova = 0
df = pd.DataFrame()
df_accuracy = pd.DataFrame()

qubits = 4
learning_rate = 0.3
dev = qml.device("default.qubit", wires = qubits)
dev_draw = qml.device('qiskit.aer', wires=qubits)
opt = qml.GradientDescentOptimizer(stepsize=learning_rate)

threshold = 5*10**(-3) #for the gradient
convergence=10**(-5) #for the VQE algorithm
params_selected = [] #values of the parameters selected in the VQE agorithm
arg_selected = [] #operators from the operator pool selected (which argument)
theta = 0.0 #initialization of the parameter used in the VQE algorithm
energy, phi = [], []

#defining the Hamiltonian of the problem
X_coefs = [1.0]*qubits
ZZ_coefs = [1.0]*qubits
TFIM_H = OP.TFIM_Hamiltonian(ZZ_coefs, X_coefs)
print('TFIM Hamiltonian: \n', TFIM_H)

#Finding the ground state of the Hamiltonian through its eigenvalues:
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

print('Dimension of the operator pool', len(operator_pool))

#sampling only some oeprators from the operator pool
n_operators = 30

seed(5)
operator_pool = sample(operator_pool, n_operators)
print('Dimension of the reduced operator pool', len(operator_pool))


#DEFINITION OF THE INITIAL STATE
params_initial = np.pi * np.random.rand(qubits, 3)
#params_initial = [[2.97696855, 3.10633172, 0.76956354], [0.46448793, 2.23320824, 2.50167315], [0.92231833, 0.77118083, 1.15315358], [2.50418158, 2.32937573, 2.23247663]]


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

    #adding the already selected operators
    for j in range(len(arg_selected)):
        apply(params_selected[j], operator_pool[arg_selected[j]])

    # adding a different operator
    apply(param, operator_pool[number])
    return qml.expval(qml.Hermitian(TFIM_H, wires=range(qubits)))

drawer = qml.draw(circuit)
result = circuit(param=0, number = 0)
print('Circuit: ')
print(drawer(param=0, number=0))
energy.append(result) #appending energy of initial state

@qml.qnode(dev)
def circuit_VQE(params, wires=range(qubits)):
    """
    Circuit definition to perform the VQE algorithm
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

    j = 0
    while (abs(energy_new - energy_old) > convergence):
        #if(j>50):
        #    break
        energy_old = energy_new
        phi = opt.step(circuit_VQE, phi)
        energy_new = circuit_VQE(phi)
        j += 1

    print('Optimized energy: ', energy_new)
    params_selected = phi
    energy.append(energy_new)

    #saving accuracy and depth of the circuit into a file
    acc = float(energy_new-ground_state)
    specs = qml.specs(circuit_VQE)(params_selected)
    depth = specs.get('depth')

    new_row= pd.Series(data={"Depth": depth, "Accuracy": acc}, name='{}'.format(i))
    df_accuracy = df_accuracy.append(new_row, ignore_index=False)


    if (energy_new < -5.22):
        break




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
specs = qnode_final.specs
print('Circuit Features: ', specs)

qnode_final_draw = qml.QNode(circuit_final, dev_draw)
result = qnode_final_draw()
dev_draw._circuit.draw(output='mpl',filename = '/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/TFIM/Circuit_TFIM_random_intial_state.pdf' )

energy_final = qnode_final()
print('Ground state energy = ', energy_final)
print('How many operators: ', len(arg_selected))
print('Used operators: ', arg_selected)
print('Parameters: ', params_selected)


df_accuracy.to_pickle("/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Depth/Circuit_depth_vs_accuracy_{}.txt".format(prova))


new_row= pd.Series(data={"Initial state":params_initial, "Energy": energy, "Learning rate": learning_rate,
                         "Operators": len(operator_pool), "Operator_pool": operator_pool,
                         "Used Operators": arg_selected, "Parameters": params_selected,
                         "Circuit features": specs, "Circuit depth": specs.get('depth')}, name='{}'.format(i))
df = df.append(new_row, ignore_index=False)
df.to_csv('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Random_state_prova_{}_{}.csv'.format(prova, len(operator_pool)))


plt.figure()
plt.title("Energy Optimization")
plt.axhline(y=ground_state, color='r', linestyle='-', label='Ground state energy')
plt.plot(energy,'o-')
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")
plt.legend()
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/TFIM/TFIM_optimization_random_initial_state.pdf')
plt.show()
