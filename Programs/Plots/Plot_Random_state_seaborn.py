#plottimg the optimization for 1 operator
#changing the dimension of the operator pool (65 vs 40, vs 30, vs 20, vs 15)
#Averaged over differnt initializations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ground_state = -5.226251859505508

energies_65 = []#energy for 65 operators in the operator pool
energies_40 = []#energy for 40 operators in the operator pool
energies_30 = []#energy for 30 operators in the operator pool
energies_20 = []#energy for 20 operators in the operator pool
energies_15 = []#energy for 15 operators in the operator pool

prove = 13
len_complete = 31

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

for i in range(prove):
    df_65 = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Random_initial_state/Random_initial_state_prova_{}_65.txt'.format(i), delimiter=',', skiprows=3, nrows=1, header=None)
    energies_65.append(df_65.values[0])

    if (len(energies_65[-1]) < len_complete):
        energies_65[-1] = adding_dimension(energies_65[-1], len_complete)

    df_40 = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Random_initial_state/Random_initial_state_prova_{}_40.txt'.format(i), delimiter=',', skiprows=3, nrows=1, header=None)
    energies_40.append(df_40.values[0])
    if (len(energies_40[-1]) < len_complete):
        energies_40[-1] = adding_dimension(energies_40[-1], len_complete)

    df_30 = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Random_initial_state/Random_initial_state_prova_{}_30.txt'.format(i), delimiter=',', skiprows=3, nrows=1, header=None)
    energies_30.append(df_30.values[0])
    if (len(energies_30[-1]) < len_complete):
        energies_30[-1] = adding_dimension(energies_30[-1], len_complete)

    df_20 = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Random_initial_state/Random_initial_state_prova_{}_20.txt'.format(i), delimiter=',', skiprows=3, nrows=1, header=None)
    energies_20.append(df_20.values[0])
    if (len(energies_20[-1]) < len_complete):
        energies_20[-1] = adding_dimension(energies_20[-1], len_complete)

    df_15 = pd.read_table('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Random_initial_state/Random_initial_state_prova_{}_15.txt'.format(i),        delimiter=',', skiprows=3, nrows=1, header=None)
    energies_15.append(df_15.values[0])
    if (len(energies_15[-1]) < len_complete):
        energies_15[-1] = adding_dimension(energies_15[-1], len_complete)


df_65 = pd.DataFrame(energies_65).melt()
df_40 = pd.DataFrame(energies_40).melt()
df_30 = pd.DataFrame(energies_30).melt()
df_20 = pd.DataFrame(energies_20).melt()
df_15 = pd.DataFrame(energies_15).melt()


sns.set()
plt.rcParams.update({'font.size': 15})
plt.axhline(y=ground_state, color='r', linestyle='-', label= 'Ground State Energy')
sns.lineplot(x="variable", y="value", data=df_65, label ='Dim 65')
sns.lineplot(x="variable", y="value", data=df_40, label ='Dim 40')
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Random_state/Mean_65_vs_40.pdf')

plt.figure()
plt.rcParams.update({'font.size': 15})
plt.axhline(y=ground_state, color='r', linestyle='-', label= 'Ground State Energy')
sns.lineplot(x="variable", y="value", data=df_65, label ='Dim 65')
sns.lineplot(x="variable", y="value", data=df_30,  label ='Dim 30')
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Random_state/Mean_65_vs_30.pdf')

plt.figure()
plt.rcParams.update({'font.size': 15})
plt.axhline(y=ground_state, color='r', linestyle='-', label= 'Ground State Energy')
sns.lineplot(x="variable", y="value", data=df_65, label ='Dim 65')
sns.lineplot(x="variable", y="value", data=df_20, label ='Dim 20')
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Random_state/Mean_65_vs_20.pdf')

plt.figure()
plt.rcParams.update({'font.size': 15})
plt.axhline(y=ground_state, color='r', linestyle='-', label= 'Ground State Energy')
sns.lineplot(x="variable", y="value", data=df_65, label ='Dim 65')
sns.lineplot(x="variable", y="value", data=df_15, label = 'Dim 15')
plt.xlabel("Iterations")
plt.ylabel("Energy, Ha")
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Random_state/Mean_65_vs_15.pdf')

