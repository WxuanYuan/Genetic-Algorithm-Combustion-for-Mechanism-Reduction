from GAPar import GeneticAlgorithmForOptimization
import multiprocessing
import os
import cantera as ct

def start_program():
    print("---  Chemical Kinetic Mechanism Optimization Using Genetic Algorithm  ---")
    print(f"Program started from root:  {current_directory}")
    print(
        "Please put all required documents under directory [source_files]. For more details about document requirements, please see [README.md].")
    print("Please input the filename for global configuration file:")
    ga = None
    while True:
        # Todo: GAinputOptimization.dat
        command = input("> ")
        try:
            ga = GeneticAlgorithmForOptimization(root_path=current_directory, parallel=True,
                                                 global_configuration_file_name=command)
            break
        except FileNotFoundError:
            print(f"At least one input file did not meet requirements.")
            print(
                f"Please follow [README.md] to check inputs files and input the filename for global configuration file again:")
            continue
    return ga


def optimization_module_command_line(ga):
    print("\n----------------------------------------")
    print("Please use one of the following commands:")
    print("[run]: start optimization")
    print("[output]: output optimized mechanism by generating an yaml file")
    print("[visualize]: show accuracy and size curve the given checkpoints by showing ")
    print("[animation]: create a GIF")
    print("[quit]: shut down program")
    print("[restart]: restart program")
    while True:
        command = input("> ")
        if command == "mute":
            ct.suppress_thermo_warnings()
            print("Thermo warnings from Cantera have been suppressed.\n")
            continue
        if command == "run":
            try:
                ga.run()
                continue
            except FileNotFoundError:
                print("Running interrupted by an error.\n")
                continue
        if command == "visualize":
            try:
                ga.visualize()
                continue
            except FileNotFoundError:
                print("Visualization interrupted by an error.\n")
                continue
        if command == "output":
            print(f"Please give the name of the checkpoint file (that locates in the [\Checkpoint] directory) for the desired output mechanism.")
            checkpoint_name = input("> ")
            print(f"Please give the prefix of the name of the output file.")
            prefix = input("> ")

            try:
                print(f"Please give the index of the output file. Type [all] to output all individuals in the checkpoint.")
                index_input = input("> ")
                if index_input == "all":
                    ga.generate_yaml_file(f"{checkpoint_name}", None, prefix)
                else:
                    try:
                        index = int(index_input)
                        ga.generate_yaml_file(f"{checkpoint_name}", index, prefix)
                    except ValueError:
                        print("Input should be either [all] or an int number.")
                # Todo: 6_RRC_final_test.csv
                continue
            except FileNotFoundError:
                print("Please check the given file name.\n")
                continue
        if command == "quit":
            return "quit"
        if command == "restart":
            print("Restarting...\n")
            return "restart"
        else:
            print("Unknown command:", command)


if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    multiprocessing.freeze_support()

    while True:
        ga = start_program()
        state = optimization_module_command_line(ga)

        if state == "quit":
            exit()
        elif state == "restart":
            continue
