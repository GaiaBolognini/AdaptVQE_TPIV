#plot the optimization obtained for the natural gardient vs the gradient descent
#averaged over differnt intializaions, taking only one oepratro at a time for complete operator pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
prove = 14
len_complete = 31
ground_state = -5.226251859505508

def adding_dimension (energy, len_compl):
    '''

    Args:
        len: dimension to be reached
        energies: vector where to apply the function (and that is changed inside the function)

    Returns:

    '''
    len_difference = len_compl - len(energy)
    add_value = energy[-1]

    list_new = [add_value]*len_difference
    energy = np.append(energy, list_new)

    return energy

energies_GD=[]
energies_QNG = []

for i in range(prove):
    df_GD = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/QNG_vs_GD/Gradient_descent_limited_VQE.txt', delimiter=',', skiprows=i, nrows=1, header=None)
    energies_GD.append(df_GD.values[0])

    if (len(energies_GD[-1]) < len_complete):
        energies_GD[-1] = adding_dimension(energies_GD[-1], len_complete)

    df_QNG = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/QNG_vs_GD/Quantum_Natural_Gradient_limited_VQE.txt', delimiter=',', skiprows=i, nrows=1, header=None)
    energies_QNG.append(df_QNG.values[0])

    if (len(energies_QNG[-1]) < len_complete):
        energies_QNG[-1] = adding_dimension(energies_QNG[-1], len_complete)

print(energies_GD)
df_GD = pd.DataFrame(energies_GD).melt()
print(df_GD)
df_QNG = pd.DataFrame(energies_QNG).melt()


sns.set()
plt.axhline(y=ground_state, color='r', linestyle='-', label= 'Ground State Energy')
sns.lineplot(x="variable", y="value", data=df_GD)
sns.lineplot(x="variable", y="value", data=df_QNG)
plt.legend(labels=['GD', 'QNG'])
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/QNG_vs_GD/QNG_vs_GD_limited_VQE.pdf')



last_values_GD, last_values_QNG = [], []
for i in range(prove):
    last_values_GD.append(energies_GD[i][-1])
    last_values_QNG.append(energies_QNG[i][-1])

last_value_GD_mean = np.mean(last_values_GD)
last_value_QNG_mean = np.mean(last_values_QNG)

print('Mean value for ground state GD: ',last_value_GD_mean)
print('Mean value for ground state QNG: ', last_value_QNG_mean)
