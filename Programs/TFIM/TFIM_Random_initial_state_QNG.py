#implementing TFIM Hamiltonian and finding its ground state:
#initializing with a random rotation where the angles are different for each qubit
#Making a comparison between the Quantum natural Gradient and the Gradient Descent for the same initialization
#allowing the VQE algorithm to run only for a limited number of steps (10)

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt
import OperatorPool_TFIM as OP
import pandas as pd
prova = 13

qubits = 4
learning_rate = 0.15
dev = qml.device("default.qubit", wires = qubits)
dev_draw = qml.device('qiskit.aer', wires=qubits)
opt_GD = qml.GradientDescentOptimizer(stepsize=learning_rate)
opt_QNG = qml.QNGOptimizer(stepsize=learning_rate)

threshold = 5*10**(-3) #for the gradient
convergence=10**(-4) #for the VQE algorithm
params_selected= []#values of the parameters selected in the VQE agorithm
arg_selected = []#operators from the operator pool selected (which argument)
theta = 0.0 #initialization of the parameter used in the VQE algorithm
energy_GD, energy_QNG = [], []

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

print('Dimension of the operator pool', len(operator_pool))

#DEFINITION OF THE INITIAL STATE
params_initial= np.pi * np.random.rand(qubits, 3)


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
result = circuit(param=0, number = 0)
print('Result: ', result)
print(drawer(param=0, number=10))

energy_GD.append(result)
energy_QNG.append(result)


#circuit for VQE algorithm
dev_prova = qml.device("default.qubit", wires=4)

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

obs = []
for i in range (qubits):
    obs.append(qml.PauliX(i))
    if i == (qubits-1):
        obs.append(qml.PauliZ(i)@qml.PauliZ(0))
    else:
        obs.append(qml.PauliZ(i)@qml.PauliZ(i+1))
coeffs = [1.0]*(2*qubits)
cost_fn = qml.ExpvalCost(circuit_VQE, qml.Hamiltonian(coeffs, obs), dev_prova)

#storing the optimized parameters
phi = np.array([], requires_grad=True)

#QUANTUM NATURAL GRADIENT
#running the VQE-Adapt algorithm
print('Quantum Natural Gradient')
for i in range(30):
    print(i)
    circuit_gradients = []

    #evaluating the gradient using the grad function
    for i in range(len(operator_pool)):
        ansatz = lambda param, number=i: circuit(param, wires=range(qubits), number=number)
        grad = qml.grad(ansatz)
        circuit_gradients.append(grad(theta))

    #taking the max value of the gradient
    max_grad = max(np.abs(circuit_gradients))

    #evaluating with respect to the threshold
    if (max_grad < threshold):
        break

    #saving the argument selected in arg_selected
    arg_selected.append(np.argmax(np.abs(circuit_gradients)))
    print(arg_selected)

    #applying the VQE to find the best set of parameters
    phi = np.append(phi, theta)#parameter initialization
    energy_old = cost_fn(phi)
    print('Initial energy: ', energy_old)

    phi = opt_QNG.step(cost_fn, phi)
    energy_new = cost_fn(phi)

    j = 0
    while (abs(energy_new - energy_old) > convergence):
        if (j>10):
            break
        energy_old = energy_new
        phi = opt_QNG.step(cost_fn, phi)
        energy_new = cost_fn(phi)
        j +=1

    print('Optimized energy: ', energy_new)
    params_selected = phi
    energy_QNG.append(energy_new)

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
specs_QNG = qnode_final.specs

energy_final = qnode_final()
print('Ground state energy = ', energy_final)
print('How many operators: ', len(arg_selected))
print('Used operators: ', arg_selected)
print('Parameters: ', params_selected)


#GRADIENT DESCENT
print('Gradient Descent')
#getting rid of previous values
arg_selected = []
params_selected = []
phi = np.array([], requires_grad=True)

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
    print(arg_selected)

    #applying the VQE to find the best set of parameters
    phi = np.append(phi, theta)#parameter initialization
    energy_old = cost_fn(phi)
    print('Initial energy: ', energy_old)

    phi = opt_GD.step(cost_fn, phi)
    energy_new = cost_fn(phi)

    j = 0
    while (abs(energy_new - energy_old) > convergence):
        if (j >10):
            break
        energy_old = energy_new
        phi = opt_GD.step(cost_fn, phi)
        energy_new = cost_fn(phi)
        j += 1

    print('Optimized energy: ', energy_new)
    params_selected = phi
    energy_GD.append(energy_new)

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

qnode_final_draw = qml.QNode(circuit_final, dev_draw)
result = qnode_final_draw()

energy_final = qnode_final()
print('Ground state energy = ', energy_final)
print('How many operators: ', len(arg_selected))
print('Used operators: ', arg_selected)
print('Parameters: ', params_selected)
specs_GD = qnode_final.specs


#saving into a file
with open("/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/QNG_vs_GD/QNG_vs_GD_initial_values_limited_VQE.txt", "a") as f:
    f.write('Prova' + str(prova) + '\n')
    np.savetxt(f, params_initial, delimiter = ',')
    f.write('\n')
    f.write('Learning-rate: ' + str(learning_rate))
    f.write('\n')
    f.write("Depth QNG: " + str(specs_QNG.get('depth')) +'\n')
    f.write('Depth GD: ' + str(specs_GD.get('depth')) +'\n')
    f.write('\n')
f.close()

energy_save = [energy_QNG]
with open("/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/QNG_vs_GD/Quantum_Natural_Gradient_limited_VQE.txt", "a") as f:
    np.savetxt(f, energy_save, delimiter = ',')
f.close()

energy_save = [energy_GD]
with open("/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/QNG_vs_GD/Gradient_descent_limited_VQE.txt", "a") as f:
    np.savetxt(f, energy_save, delimiter = ',')
f.close()

#plotting
plt.figure()
plt.title("Energy Optimization")
plt.axhline(y=ground_state, color='r', linestyle='-', label='Ground state energy')
plt.plot(energy_GD, 'o-', label = 'Gradient Descent')
plt.plot(energy_QNG,'o-', label = 'Quantum Natural Gradient')
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")
plt.legend()
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/TFIM/TFIM_optimization_random_initial_state_QNG_vs_GD.pdf')
plt.show()
