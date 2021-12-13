#plotting the optimization for combinations of two operators vs single operators for 20 operators in the operator pool
#taking the mean value and the variance band with seaborn
#If I want also the NGD it is enough to change the name of the files in df_combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ground_state = -5.226251859505508
energies = []
energies_combinations = []
prove = 13
len_complete = 31

#adding dimension: if dim<31 add the last element
def adding_dimension (energy, len_compl):
    '''

    Args:
        len: dimension to be reached
        energies: vector where to apply the function (and that is changed inside the function)

    Returns: new vector with new dimension

    '''
    len_difference = len_compl - len(energy)
    add_value = energy[-1]

    list_new = [add_value]*len_difference
    energy = np.append(energy, list_new)

    return energy

for prova in range(prove):
    df = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Combinations2_and_singles/Combinations2_and_singles_{}_20_QNG.txt'.format(prova), delimiter=',', skiprows=3, nrows=1, header=None)
    energies.append(df.values[0])

    #adding dimension
    if (len(energies[-1]) < len_complete):
        energies[-1] = adding_dimension(energies[-1], len_complete)

    df_combinations = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Combinations2_and_singles/Combinations2_and_singles_{}_20_divided2.txt'.format(prova), delimiter=',', skiprows=3, nrows=1, header=None)
    energies_combinations.append(df_combinations.values[0])

    if (len(energies_combinations[-1]) < len_complete):
        energies_combinations[-1] = adding_dimension(energies_combinations[-1], len_complete)


df = pd.DataFrame(energies).melt()
df_combinations = pd.DataFrame(energies_combinations).melt()


sns.set()
plt.axhline(y=ground_state, color='r', linestyle='-', label= 'Ground State Energy')
sns.lineplot(x="variable", y="value", data=df_combinations, label = 'GD')
sns.lineplot(x="variable", y="value", data=df, label='QNG')
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Combinations/Combinations_vs_singles_QNG.pdf')

