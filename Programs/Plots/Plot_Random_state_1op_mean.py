#plotting optimization process for different dimension of the operator pool.
#taking the mean value over different intializations. 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ground_state = -5.226251859505508
energies_65 = []#energy for 65 operators in the operator pool
energies_40 = []#energy for 40 operators in the operator pool
energies_30 = []#energy for 30 operators in the operator pool
energies_20 = []#energy for 20 operators in the operator pool
energies_15 = []#energy for 15 operators in the operator pool

prove = 13

for i in range(prove):
    df_65 = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Random_initial_state/Random_initial_state_prova_{}_65.txt'.format(i), delimiter=',', skiprows=3, nrows=1, header=None)
    energies_65.append(df_65.values[0])
    df_40 = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Random_initial_state/Random_initial_state_prova_{}_40.txt'.format(i), delimiter=',', skiprows=3, nrows=1, header=None)
    energies_40.append(df_40.values[0])
    df_30 = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Random_initial_state/Random_initial_state_prova_{}_30.txt'.format(i), delimiter=',', skiprows=3, nrows=1, header=None)
    energies_30.append(df_30.values[0])
    df_20 = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Random_initial_state/Random_initial_state_prova_{}_20.txt'.format(i), delimiter=',', skiprows=3, nrows=1, header=None)
    energies_20.append(df_20.values[0])
    df_15 = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Random_initial_state/Random_initial_state_prova_{}_15.txt'.format(i),        delimiter=',', skiprows=3, nrows=1, header=None)
    energies_15.append(df_15.values[0])

#adding dimensions if the vector has not been completed: we take the final value repeated
#indeed the cycle stopped when the gradient was too little: we can suppose of having reached the minimum for that configuration
def adding_dimension (energies, len_compl):
    '''

    Args:
        len: dimension to be reached
        energies: vector where to apply the function (and that is changed inside the function)

    Returns:

    '''

    for i in range(len(energies)):
        len_difference = len_compl - len(energies[i])
        add_value = energies[i][-1]
        list_new = [add_value]*len_difference
        energies[i] = np.append(energies[i], list_new)

    return


len_complete = 31
adding_dimension(energies_65, len_complete)
adding_dimension(energies_40, len_complete)
adding_dimension(energies_30, len_complete)
adding_dimension(energies_20, len_complete)
adding_dimension(energies_15, len_complete)


#computing the mean value for the different vector
energies_65_mean = np.mean(energies_65, axis=0)
energies_40_mean = np.mean(energies_40, axis=0)
energies_30_mean = np.mean(energies_30, axis=0)
energies_20_mean = np.mean(energies_20, axis=0)
energies_15_mean = np.mean(energies_15, axis=0)

print('15: ',energies_15_mean[-1])
print('20: ',energies_20_mean[-1])
print('30: ',energies_30_mean[-1])
print('40: ',energies_40_mean[-1])
print('65: ',energies_65_mean[-1])

plt.figure()
plt.axhline(y=ground_state, color='r', linestyle='-', label= 'Ground State Energy')
plt.plot(energies_65_mean, label = 'Dim = 65')
plt.plot(energies_40_mean, label = 'Dim = 40')
plt.plot(energies_30_mean, label = 'Dim = 30')
plt.plot(energies_20_mean, label = 'Dim = 20')
plt.plot(energies_15_mean, label = 'Dim = 15')
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")
plt.legend()
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Random_state/Mean_values.pdf')
plt.show()


