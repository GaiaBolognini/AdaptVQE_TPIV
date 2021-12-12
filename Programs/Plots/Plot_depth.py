#plotting the error of the energy vs the circuit depth
#averaged over many trials (mean value and variance band)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
prove = 13
len_compl = 31

def adding_dimension (value, len_compl):
    '''

    Args:
        len: dimension to be reached
        energies: vector where to apply the function (and that is changed inside the function)

    Returns:

    '''
    len_difference = len_compl - len(value)
    add_value = value[-1]

    list_new = [add_value]*len_difference
    value = np.append(value, list_new)

    return value

accuracy_mean, accuracy_dev= [0]*len_compl, [0]*len_compl
depth_mean, depth_dev = [0]*len_compl, [0]*len_compl

for i in range(prove):
    df = pd.read_pickle("/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Results/Depth/Circuit_depth_vs_accuracy_{}.txt".format(i))
    accuracy = df.values[:,0]
    accuracy = adding_dimension(accuracy, len_compl)
    accuracy_mean += accuracy
    accuracy_dev +=accuracy*accuracy

    depth = df.values[:,1]
    depth = adding_dimension(depth, len_compl)
    depth_mean += depth
    depth_dev += depth * depth

accuracy_mean = accuracy_mean/prove
accuracy_dev = np.abs(accuracy_dev/prove - accuracy_mean*accuracy_mean)
accuracy_dev = np.sqrt(accuracy_dev)

depth_mean = depth_mean/prove
depth_dev = depth_dev/prove - depth_mean*depth_mean
depth_dev = np.sqrt(depth_dev)

sns.set()
plt.plot(depth_mean, accuracy_mean)
plt.fill_between(depth_mean, accuracy_mean-accuracy_dev, accuracy_mean+accuracy_dev, color='skyblue')
plt.fill_betweenx(accuracy_mean, depth_mean-depth_dev, depth_mean+depth_dev, color='skyblue')
plt.title("Error vs Depth, Averages")
plt.xlabel("Depth")
plt.ylabel("Error")
plt.savefig('/mnt/c/Users/gaias/Desktop/Adapt_VQE_TPIV/Images/Depth/Accuracy_depth_errors.pdf')

