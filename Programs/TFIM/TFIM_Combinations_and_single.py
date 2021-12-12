# problem: TFIM Hamiltonian
# initializing the state randomly
# taking combinations of two operators and single operators in the Adapt-VQE
#Possible to choose if to use the Quantum natural Gradient or the Gradient descent ot perform the VQE algorithm
#Changing the optimizer is prbably necessary to change the learning rate


import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt
import OperatorPool_TFIM as OP
import itertools
from random import sample
from random import seed

prova = 1

qubits = 4
dev = qml.device("default.qubit", wires=qubits)
learning_rate = 0.3
dev_draw = qml.device('qiskit.aer', wires=qubits)
opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
#opt = qml.QNGOptimizer(stepsize = learning_rate)


threshold = 5 * 10 ** (-2)  # for the gradient
convergence = 10**(-5)  # for the VQE algorithm
params_selected = []  # values of the parameters selected in the VQE agorithm
#arg selected are chosen is such a way that if arg_selected<len(operator_pool), it is a single operator
#otherwise it is a combination
arg_selected = []  # operators from the operator pool selected (which argument)
energy = []

# defining the Hamiltonian of the problem
X_coefs = [1.0] * qubits
ZZ_coefs = [1.0] * qubits
TFIM_H = OP.TFIM_Hamiltonian(ZZ_coefs, X_coefs)

# finding the eigenvalues of the Hamiltonian:
eigenvals, eigenvects = np.linalg.eigh(TFIM_H)
ground_state = min(np.real(eigenvals))
print('Ground state energy: ', ground_state)


# DEFINITION OF THE OPERATOR POOL
def apply(params, operator):
    return operator(params)
operator_pool = []

# Operator pool from H
operator_pool = OP.making_operator_pool(qubits, operator_pool)

# creating H^2 operator pool
OP.Operator_XX(qubits, operator_pool)
OP.Operator_ZZ_notNN(qubits, operator_pool)
OP.Operator_ZY(qubits, operator_pool)
OP.Operator_ZZX(qubits, operator_pool)
OP.Operator_ZZZZ(qubits, operator_pool)

# new operator defintion (not belonging to the Hamiltonian)
OP.Operator_XY(qubits, operator_pool)
OP.Operator_ZX(qubits, operator_pool)
OP.Operator_YY(qubits, operator_pool)
OP.Operator_ZXY(qubits, operator_pool)

#sampling from the operator pool
n_operators = 20
seed(5)
operator_pool = sample(operator_pool, n_operators)
print('Single operators:', len(operator_pool))

# defining all the possible combinations in the operator pools
n_combinations = 2  # number of elements in the combination
combinations = itertools.combinations(range(len(operator_pool)), n_combinations)
combinations = [i for i in combinations]
print('Combinations: ', combinations)
print('Number of possible combinations: ', len(combinations))

# DEFINITION OF THE INITIAL STATE
params_initial_rnd = np.pi * np.random.rand(qubits, 3)
params_initial = np.copy(params_initial_rnd)

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
    for i in range(qubits):
        angles = params_initial[i][:]
        qml.Rot(angles[0], angles[1], angles[2], wires=i)

    # adding the already selected operators
    index = 0 #needed to keep track of the correspondence arg_selected-params_selected (any time a param is selected add 1)
    for j in range(len(arg_selected)):
        if (arg_selected[j] < len(operator_pool)):
            #then apply a single operator
            apply(params_selected[index], operator_pool[arg_selected[j]])
            index += 1
        else:
            #apply a combination
            apply_operators = combinations[arg_selected[j]-len(operator_pool)]
            for i in range(n_combinations):
                apply(params_selected[index], operator_pool[apply_operators[i]])
                index += 1

    #adding a different operator
    if (len(number) == n_combinations):
        for i in range(n_combinations):
            apply(param[i], operator_pool[number[i]])
    else:
        apply(param, operator_pool[number[0]])

    return qml.expval(qml.Hermitian(TFIM_H, wires=range(qubits)))


drawer = qml.draw(circuit)
result = circuit(param=[0,0], number=combinations[0])
print('Circuit: ')
print(drawer(param=[0,0], number=combinations[0]))
energy.append(result)

# circuit for VQE algorithm
dev_prova = qml.device("default.qubit", wires=4)

def circuit_VQE(params, wires=range(qubits)):
    """
    Circuit definition to compute the VQE algorithm
    Returns: Expectation value of the Hamiltonian
    """
    for i in range(qubits):
        angles = params_initial[i][:]
        qml.Rot(angles[0], angles[1], angles[2], wires=i)

    # adding the chosen operators
    index = 0    #needed to keep track of the correspondence arg_selected-params_selected (any time a param is selected add 1)
    for j in range(len(arg_selected)):
        if (arg_selected[j] < len(operator_pool)):
            #then apply a single operator
            apply(params[index], operator_pool[arg_selected[j]])
            index += 1
        else:
            #apply a combination
            apply_operators = combinations[arg_selected[j]-len(operator_pool)]
            for i in range(n_combinations):
                apply(params[index], operator_pool[apply_operators[i]])
                index += 1

obs = []
for i in range(qubits):
    obs.append(qml.PauliX(i))
    if i == (qubits - 1):
        obs.append(qml.PauliZ(i) @ qml.PauliZ(0))
    else:
        obs.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
coeffs = [1.0] * (2 * qubits)
cost_fn = qml.ExpvalCost(circuit_VQE, qml.Hamiltonian(coeffs, obs), dev_prova)

# storing the optimized parameters
phi = np.array([])

# running the VQE-Adapt algorithm
for i in range(30):

    '''
    #in the QNG it is sometimes necessary to update the learning rate through the simulation
    if (i > 10):
         learning_rate = 0.05
         opt = qml.QNGOptimizer(stepsize=learning_rate)
    '''

    print(i)
    gradient_lenghts = []  # storing the lenght of the gradient

    #evaluating the gradient with respect to single operators
    for i in range(len(operator_pool)):
        theta = [0.0]
        ansatz = lambda param, number=[i]: circuit(param, wires=range(qubits), number=number)
        grad = qml.grad(ansatz)
        gradient_lenghts.append(np.abs(grad(theta)))

    # evaluating the gradient with respect to combinations of operators
    for i in range(len(combinations)):
        theta = np.zeros((n_combinations))
        ansatz = lambda param, number=combinations[i]: circuit(param, wires=range(qubits), number=number)
        grad = qml.grad(ansatz)
        gradient_eval = grad(theta)
        # taking the norm of the gradient
        norm = np.linalg.norm(gradient_eval)

        gradient_lenghts.append(np.sqrt(norm)/2)


    # taking the operator or the combination with maximum gradient norm
    max_norm = max(np.abs(gradient_lenghts))
    print('Maximum lenght',max_norm)

    # evaluating with respect to the threshold
    if (max_norm < threshold):
        break

    #saving the argument selected in arg_selected
    arg_selected.append(np.argmax(gradient_lenghts))

    if (arg_selected[-1] < len(operator_pool)):
        theta = [0.0]
    else:
        theta = np.zeros((n_combinations))

    # applying the VQE to find the best set of parameters
    phi = np.append(phi, theta)  # parameter initialization
    energy_old = cost_fn(phi)
    print('Initial energy: ', energy_old)

    phi = opt.step(cost_fn, phi)
    energy_new = cost_fn(phi)

    j = 0
    while (abs(energy_new - energy_old) > convergence):
        if (j > 50):
            break
        energy_old = energy_new
        phi = opt.step(cost_fn, phi)
        energy_new = cost_fn(phi)
        j += 1

    print('Optimized energy: ', energy_new)
    params_selected = phi
    energy.append(energy_new)

    if (energy_new < -5.22):
        break


# building the final circuit and evaluating the final energy
def circuit_final(wires=range(qubits)):
    """
    Final circuit: built with the selected operators and parameters
    Returns:Expectation value of the Hamiltonian
    """
    for i in range(qubits):
        angles = params_initial[i][:]
        qml.Rot(angles[0], angles[1], angles[2], wires=i)

    # adding the chosen operators
    index = 0  # needed to keep track of the correspondence arg_selected-params_selected (any time a param is selected add 1)
    for j in range(len(arg_selected)):
        if (arg_selected[j] < len(operator_pool)):
            # then apply a single operator
            apply(params_selected[index], operator_pool[arg_selected[j]])
            index += 1
        else:
            # apply a combination
            apply_operators = combinations[arg_selected[j] - len(operator_pool)]
            for i in range(n_combinations):
                apply(params_selected[index], operator_pool[apply_operators[i]])
                index += 1

    return qml.expval(qml.Hermitian(TFIM_H, wires=range(qubits)))


qnode_final = qml.QNode(circuit_final, dev)
drawer = qml.draw(qnode_final)
print('Circuit: ')
print(drawer())
specs = qnode_final.specs
print(specs)

qnode_final_draw = qml.QNode(circuit_final, dev_draw)
result = qnode_final_draw()
dev_draw._circuit.draw(output='mpl',filename='/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Combinations/Circuit_TFIM_2combinations_and_singles.pdf')

#saving results into a file
energy_save = [energy]
with open("/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Combinations2_and_singles/Combinations2_and_singles_{}_{}_divided2.txt".format(prova, len(operator_pool)), "w") as f:
    f.write('Initial state:\n' + str(params_initial))
    f.write('\n')
    f.write('Energies: \n')
    np.savetxt(f, energy_save, delimiter=',')
    f.write('Learning rate: ' + str(learning_rate))
    f.write('\n')
    f.write('Operators: ' + str(len(operator_pool)))
    f.write('\n')
    f.write('Combinations: ' + str(len(combinations)))
    f.write('\n')
    f.write('Operator Pool: ' + str(operator_pool))
    f.write('\n')
    f.write('Used operators #' + str(len(arg_selected)) + ' : ' + str(arg_selected))
    f.write('\n')
    f.write('Parameters: \n')
    np.savetxt(f, params_selected, delimiter = ',')
    f.write('\n')
    f.write('#Combinations: '+ str(len(params_selected) - len(arg_selected)))
    f.write('\n')
    f.write('Circuit Features: \n')
    f.write(str(specs))
    f.write('\n')
    f.write('Circuit Depth: \n')
    f.write(str(specs.get('depth')))
f.close()

energy_final = qnode_final()
print('Ground state energy = ', energy_final)
print('How many operators: ', len(arg_selected))
print('Used operators: ', arg_selected)
print('Parameters: ', params_selected)

plt.figure()
plt.title("Energy Optimization")
plt.axhline(y=ground_state, color='r', linestyle='-', label='Ground state energy')
plt.plot(energy, 'o-')
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")
plt.legend()
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Combinations/Energy_combinations_and_singles.pdf')
plt.show()
