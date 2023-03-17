import time
import numpy as np
import cantera as ct
from GAPar import GeneticAlgorithmForOptimization, GeneticAlgorithmForReduction
import multiprocessing
import os
from Data.gas import GasForOptimization, GasForReduction


def reactants_and_products(reaction):
    return list(reaction.products.keys()) + list(reaction.reactants.keys())


if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print(current_directory)
    multiprocessing.freeze_support()
    print("Program has started!")

    # ga = GeneticAlgorithmForOptimization(root_path=current_directory, parallel=True)
    # ga.generated_yaml_file(0)

    #########################################

    ga = GeneticAlgorithmForReduction(root_path=current_directory, parallel=True)
    # ga.generated_yaml_file(0)
    # ga.visualize("Reduction", 50)
    # ga.makeGrid("Reduction", 50)
    ga.print_IDT_results(1)
    print("Program has ended, this window will be closed in 10 seconds.")
    time.sleep(10)

# import matplotlib.pyplot as plt
#
# matrix = np.loadtxt('IDT.csv', delimiter=',')
# matrix = np.log10(matrix)
# markers = ['o', '+', '*', '.', ]
# markeredgecolors = ['b', 'r', 'y', 'g', ]
# labels = ['experimental data', 'detailed mech', 'output mech', 'less accuracy mech', ]
#
# for i in range(0):
#     plt.plot(np.arange(len(matrix[:, i])), matrix[:, i], label=labels[i], linestyle='dashed', linewidth=1.1, marker=markers[i], markersize=4, markeredgecolor=markeredgecolors[i])

# plt.legend(loc='upper left')
# plt.show()