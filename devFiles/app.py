# import time
# import numpy as np
# import cantera as ct
# from GAPar import GeneticAlgorithmForOptimization
# import multiprocessing
# import os
#
#
# if __name__ == '__main__':
#     current_directory = os.path.dirname(os.path.abspath(__file__))
#     print(current_directory)
#     multiprocessing.freeze_support()
#     print("Program has started!")
#
#     ga = GeneticAlgorithmForOptimization(root_path=current_directory)
#     ga.run()
#     print("Program has ended, this window will be closed in 10 seconds.")
#     time.sleep(10)

import datetime

# get the current time
current_time = datetime.datetime.now()

# format the time as "HH:MM:SS"
formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")

# print the formatted time
print("Current Time =", formatted_time)
