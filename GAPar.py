import pandas as pd
import numpy as np
from multiprocessing import Queue, Process
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cantera as ct
from Data.gas import GasForOptimization, GasForReduction
from Data.learnable_parameters import LearnableParameter, ReactionRateConstants, ReductionCode
from Data.Labels import PFR_Label, IDT_Label
from abc import abstractmethod, ABCMeta
import imageio
from Tools.tools import compute_IDP_error, compute_PFR_error, compute_sensitivity, binary_permutation
import datetime

class BaseGA(metaclass=ABCMeta):
    '''
    This is an abstract class that defines the operators to be implement.
    '''

    def __init__(self, root_path, early_stop=None, test_GA=False,
                 parallel=True, checkpoint_path=None, max_iteration=None):
        '''
        The program consists of the following components:
            (a) A Gas object, a interface to Cantera, which produces the mechanism object for simulation and evaluation
            (b) A LearnableParameter object, which stores and encodes the optimization targets. GA operators are also implemented in this class.
            (c) A Label object, which stores experimental data.

        :param root_path: root path of the program
        :param early_stop: early stop or not
        :param test_GA: test mode
        :param parallel: multi-processing
        :param checkpoint_path: path for the checkpoint file
        :param max_iteration: maximum iterations
        :return: None
        '''
        self.root_path = root_path
        self.test_GA = test_GA
        self.parallel = parallel
        self.max_iteration = max_iteration

        try:
            def read_initial_population(file_name):
                chrom = pd.read_csv(f"Checkpoints\\{file_name}")
                return chrom.values[:, 1:]

            self.initial_population = read_initial_population(checkpoint_path)
            print(f"Checkpoint loaded from path [Checkpoints\\{checkpoint_path}].")
        except FileNotFoundError:
            print("No checkpoint loaded. Start with random population.")
            self.initial_population = None

        self.early_stop = early_stop

        self.learnable_parameters = None
        self.define_chromosome()
        if not isinstance(self.learnable_parameters, LearnableParameter):
            print(f"Unknown learnable parameters type: {type(self.learnable_parameters)}")
            raise RuntimeError

        self.idp_Label = None
        self.pfr_Label = None
        self.load_experience_data()
        if self.pfr_Label is not None:
            if not isinstance(self.idp_Label, IDT_Label):
                print(f"Unknown experienced data type: {type(self.idp_Label)}")
                raise RuntimeError
        if self.pfr_Label is not None:
            if not isinstance(self.pfr_Label, PFR_Label):
                print(f"Unknown experienced data type: {type(self.pfr_Label)}")
                raise RuntimeError

        self.previous_learnable_parameters = None
        self.FitV = None

    @abstractmethod
    def ranking(self):
        raise NotImplementedError

    @abstractmethod
    def runtime_record(self):
        raise NotImplementedError

    @abstractmethod
    def save_parents(self):
        raise NotImplementedError

    def selection(self):
        self.learnable_parameters.selection(self.FitV)

    def crossover(self):
        self.learnable_parameters.crossover()

    def mutation(self):
        self.learnable_parameters.mutation()

    @abstractmethod
    def save_to_local(self, iteration_index):
        raise NotImplementedError

    @abstractmethod
    def plot_results(self):
        raise NotImplementedError

    @abstractmethod
    def initialize_population(self):
        raise NotImplementedError

    @abstractmethod
    def define_chromosome(self):
        raise NotImplementedError

    @abstractmethod
    def load_experience_data(self):
        raise NotImplementedError

    def run(self):
        '''
        Run the GA.
        '''
        self.initialize_population()
        for i in range(self.max_iteration):
            print(f"Start iteration {i}.")
            self.ranking()
            self.save_to_local(iteration_index=i)
            self.runtime_record()
            print(f"#################### Error before iteration {i + 1}: {-np.average(self.FitV)} ####################")
            self.save_parents()
            self.selection()
            self.crossover()
            self.mutation()
        self.plot_results()


class GeneticAlgorithmForOptimization(BaseGA):
    '''
    This is the control unit for RRC optimization tasks.
    '''

    def __init__(self, root_path, early_stop=None, test_GA=False, parallel=True, global_configuration_file_name="\GAinputOptimization.dat"):
        '''
        The program consists of the following components:
            (a) A Gas object, a interface to Cantera, which produces the mechanism object for simulation and evaluation
            (b) A LearnableParameter object, which stores and encodes the optimization targets. GA operators are also implemented in this class.
            (c) A Label object, which stores experimental data.

        :param root_path: root path of the program
        :param early_stop: early stop or not
        :param test_GA: test mode
        :param parallel: multi-processing
        :return: None
        '''

        # GA hyper-parameter configurations
        try:
            with open(root_path + "\source_files\\" + global_configuration_file_name, 'r') as input_data:
                self.dimension = int((input_data.readline()).split(' ', 1)[0])  # Dimension
                self.mutation_rate = float((input_data.readline()).split(' ', 1)[0])  # Mutation rate
                self.population = int((input_data.readline()).split(' ', 1)[0])  # Population size
                self.max_iteration = int((input_data.readline()).split(' ', 1)[0])  # Maximum iteration
                self.initial_iteration = int((input_data.readline()).split(' ', 1)[0])  # initial iteration
                self.precision = float((input_data.readline()).split(' ', 1)[0])  # precision
                self.max_process_number = int((input_data.readline()).split(' ', 1)[0])  # Maximum process
                self.checkpoint_path = (input_data.readline()).split(' ', 1)[0]  # checkpoint_path
                self.mechanism_yaml_path = (input_data.readline()).split(' ', 1)[0]  # mechanism yaml path
                self.IDT_configuration_path = (input_data.readline()).split(' ', 1)[0]  # IDT configuration path
                self.experimental_IDT_path = (input_data.readline()).split(' ', 1)[0]  # experimental IDT path
                self.PFR_configuration_path = (input_data.readline()).split(' ', 1)[0]  # PFR configuration path
                self.experimental_PFR_path = (input_data.readline()).split(' ', 1)[0]  # experimental PFR path
                self.optimization_pointers_file = (input_data.readline()).split(' ', 1)[0]  # optimization_pointers_file
                self.optimization_interval = (input_data.readline()).split(' ', 1)[0]  # optimization interval
                self.output_file_name = (input_data.readline()).split(' ', 1)[0]  # output_file_name
            print(f"Successfully loaded Global configuration file.")
        except FileNotFoundError:
            print(f"Global configuration file NOT found under: \\source_files\\{global_configuration_file_name}")
            raise FileNotFoundError


        super().__init__(root_path, early_stop, test_GA, parallel, self.checkpoint_path, self.max_iteration)
        # global record
        self.global_best_X = np.zeros(self.dimension)
        self.global_best_learnable_parameter_values = np.zeros(self.dimension)
        self.global_lowest_error = 100.0
        self.best_X_of_generations = []
        self.best_learnable_parameters_values_of_generations = []
        self.average_error_of_generations = []
        self.errors_of_generations = []

    def load_experience_data(self):
        '''
        load experienced data.
        '''
        # labels and learnable parameters
        self.idp_Label = IDT_Label(self.dimension, self.root_path, self.IDT_configuration_path, self.experimental_IDT_path)
        try:
            self.pfr_Label = PFR_Label(self.dimension, self.root_path, self.PFR_configuration_path, self.experimental_PFR_path)
        except FileNotFoundError:
            self.pfr_Label = None
            print("PFR error ignored.")

    def define_chromosome(self):
        '''
        Define the optimization targets (RCC here) and its encoding.
        '''
        self.learnable_parameters = ReactionRateConstants(self.root_path, self.population, self.initial_population)

    def compute_error_for_one_gene(self, index):
        """ Perform evaluation without multi-processing.
        """
        queue = Queue()
        p = Process(target=compute_error_for_Optimization, args=(self, index, queue))
        p.start()
        p.join()
        index, error = queue.get()
        return error

    def initialize_population(self):
        """
        Initialize the optimization targets (RCC here) and its encoding.
        """
        self.learnable_parameters.initialize_population()

    def ranking(self):
        """
        Perform evaluation with multi-processing.
        """
        # compute error
        start_time = time.time()
        if not self.parallel:
            errors = np.array([self.compute_error_for_one_gene(i) for i in tqdm(range(self.population))])
        else:
            queue = Queue()
            process_list = []
            total_process_count = 0
            for i in range(self.max_process_number):
                p = Process(target=compute_error_for_Optimization, args=(self, total_process_count, queue))
                total_process_count += 1
                p.start()
                process_list.append(p)

            index_and_errors = []
            for i in tqdm(range(self.population)):
                index_and_errors.append(queue.get())
                if total_process_count < self.population:
                    p = Process(target=compute_error_for_Optimization, args=(self, total_process_count, queue))
                    total_process_count += 1
                    p.start()
                    process_list.append(p)

            def sort_by_index(t):
                return t[0]

            sorted_index_and_errors = sorted(index_and_errors, key=sort_by_index)
            errors = []
            for i in range(self.population):
                errors.append(sorted_index_and_errors[i][1])
            errors = np.array(errors)

        # GA select the one with the biggest ranking, but we want to minimize the error, so we put a negative here
        self.FitV = -errors
        end_time = time.time()
        print(f"Took time {end_time - start_time} seconds.")

    def save_parents(self):
        """
        Save the learnable parameters in self.previous_learnable_parameters.
        """
        previous_learnable_parameters = ReactionRateConstants(self.root_path, self.population,
                                                              self.learnable_parameters.chrom)
        self.previous_learnable_parameters = previous_learnable_parameters

    def runtime_record(self):
        """
        Save the chromosomes and fitness values after each iteration.
        """
        best_gene_index_of_generation = np.argmax(self.FitV)
        best_X_of_generation = self.learnable_parameters.X[best_gene_index_of_generation]
        best_learnable_parameters_values_of_generation = self.learnable_parameters.get_real_values()[
            best_gene_index_of_generation]
        average_error_of_generation = -np.average(self.FitV)

        if -self.FitV[best_gene_index_of_generation] < self.global_lowest_error:
            self.global_lowest_error = -self.FitV[best_gene_index_of_generation]
            self.global_best_X = self.learnable_parameters.X[best_gene_index_of_generation]
            self.global_best_learnable_parameter_values = self.learnable_parameters.get_real_values()[
                best_gene_index_of_generation]

        self.best_X_of_generations.append(best_X_of_generation)
        self.best_learnable_parameters_values_of_generations.append(best_learnable_parameters_values_of_generation)
        self.average_error_of_generations.append(average_error_of_generation)
        self.errors_of_generations.append(-self.FitV)

    def save_to_local(self, iteration_index):
        """
        Save the chromosomes and fitness values to checkpoint files.
        """
        self.learnable_parameters.save_chrom2checkpoint(iteration_index, self.initial_iteration, self.output_file_name, fitness=self.FitV)

    def plot_results(self):
        """
        Visualize fitness values.
        """
        # Plt the average error and lowest error curve of this runtime.
        Y_history = pd.DataFrame(self.errors_of_generations)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.show()

        column1 = np.reshape(self.global_best_X, (54, 1))
        column2 = np.reshape(self.global_best_learnable_parameter_values, (54, 1))
        matrix = np.hstack([column1, column2])
        s1 = pd.DataFrame(matrix, columns=['X', 'values'])
        s1.to_excel('Best_result_RRC.xlsx', index=False, header=False)

    def generate_yaml_file(self, checkpoint_path_for_output, index, prefix):
        """
        Create a Cantera file for the mechanism described by the [index]-th chromosome in the current population.
        """
        try:
            def read_initial_population(file_name):
                chrom = pd.read_csv(f"Checkpoints\\{file_name}")
                return chrom.values[:, 1:]

            output_population = read_initial_population(checkpoint_path_for_output)
            print(f"Checkpoint file loaded from path [Checkpoints\\{checkpoint_path_for_output}].")
        except FileNotFoundError:
            print(f"No checkpoint file loaded from path [Checkpoints\\{checkpoint_path_for_output}].")
            raise FileNotFoundError
        assert len(output_population) == self.population, "Please set population equal to number of individuals in the checkpoint file."
        output_learnable_parameters = ReactionRateConstants(self.root_path, self.population, output_population)
        output_learnable_parameters.initialize_population()

        if index is None:
            for i in range(self.population):
                output_gas = GasForOptimization(output_learnable_parameters, i, self.previous_learnable_parameters,
                                         self.mechanism_yaml_path, self.optimization_pointers_file)
                string = output_gas.get_gas(return_description=True)
                output_file_name = f"output\\{prefix}_individual{i}.yaml"
                f = open(output_file_name, mode="w")
                print(f"Opimized mechanism saved to [{output_file_name}].\n")
                f.write(string)
                f.close()
        else:
            output_gas = GasForOptimization(output_learnable_parameters, index, self.previous_learnable_parameters,
                                     self.mechanism_yaml_path, self.optimization_pointers_file)
            string = output_gas.get_gas(return_description=True)
            output_file_name = f"output\\{prefix}_individual{index}.yaml"
            f = open(output_file_name, mode="w")
            print(f"Opimized mechanism saved to [{output_file_name}].\n")
            f.write(string)
            f.close()

    def visualize(self):
        """
        Read checkpoint files and draw the accuracy curve and size curve.
        """
        fitVs = []
        i = 0
        while True:
            try:
                content = pd.read_csv(f"Checkpoints/{i}_{self.output_file_name}").to_numpy()
                fitVs.append(np.max(content[:, -1]))
                i += 1
            except FileNotFoundError:
                break

        if len(fitVs) == 0:
            raise FileNotFoundError

        # Plt the average error and lowest error curve of this runtime.
        fig = plt.figure()
        plt.plot(np.arange(len(fitVs)), fitVs, '.', color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Values')
        plt.title("Fitness value of the best chromosome in each iteration.")
        plt.show()


def compute_error_for_Optimization(ga: GeneticAlgorithmForOptimization, index, queue):
    """ Compute the error that consists of IDP_error and PFR_error.
    """
    # print(f"Ranking for gene {index}.")
    if ga.test_GA:
        error = np.sum(ga.learnable_parameters.X[index])
    else:
        ct.suppress_thermo_warnings()
        gas = GasForOptimization(ga.learnable_parameters, index, ga.previous_learnable_parameters,
                                 ga.mechanism_yaml_path, ga.optimization_pointers_file)
        IDP_error = compute_IDP_error(ga.idp_Label, gas.get_gas())
        if ga.pfr_Label is None:
            PFR_error = 0.0
        else:
            PFR_error = compute_PFR_error(ga.pfr_Label, gas.get_gas())
        error = (IDP_error + PFR_error) / 2
    queue.put((index, error))


class GeneticAlgorithmForReduction(BaseGA):
    '''
    This is the control unit for mechanism reduction tasks.
    '''

    def __init__(self, root_path, early_stop=None, test_GA=False, parallel=True,
                 global_configuration_file_name="GAinputReduction.dat"):
        '''
        The program consists of the following components:
            (a) A Gas object, a interface to Cantera, which produces the mechanism object for simulation and evaluation
            (b) A LearnableParameter object, which stores and encodes the optimization targets. GA operators are also implemented in this class.
            (c) A Label object, which stores experimental data.

        :param root_path: root path of the program
        :param early_stop: early stop or not
        :param test_GA: test mode
        :param parallel: multi-processing
        :return: None
        '''

        # GA hyper-parameter configurations
        try:
            print(f"Loading global configuration file.")
            with open(root_path + "\source_files" + f"\{global_configuration_file_name}", 'r') as input_data:
                self.n_species = int((input_data.readline()).split(' ', 1)[0])  # number of species
                self.n_reactions = int((input_data.readline()).split(' ', 1)[0])  # number of reactions
                self.mutation_rate = float((input_data.readline()).split(' ', 1)[0])  # Mutation rate
                self.population = int((input_data.readline()).split(' ', 1)[0])  # Population size
                self.max_iteration = int((input_data.readline()).split(' ', 1)[0])  # Maximum iteration
                self.initial_iteration = int((input_data.readline()).split(' ', 1)[0])  # initial iteration
                self.max_process_number = int((input_data.readline()).split(' ', 1)[0])  # Maximum process
                self.mechanism_yaml_path = (input_data.readline()).split(' ', 1)[0]  # mechanism yaml path
                self.IDT_configuration_path = (input_data.readline()).split(' ', 1)[0]  # IDT configuration path
                self.experimental_IDT_path = (input_data.readline()).split(' ', 1)[0]  # experimental IDT path
                self.checkpoint_path = (input_data.readline()).split(' ', 1)[0]  # checkpoint_path
                self.sensitivities_file_name = (input_data.readline()).split(' ', 1)[0]  # sensitivities file name
                self.output_file_name = (input_data.readline()).split(' ', 1)[0]  # output_file_name
                self.delta = float((input_data.readline()).split(' ', 1)[0])  # delta
                self.important_species = (input_data.readline()).split(' ')  # important-species
                self.important_species = [n.replace(',', '') for n in self.important_species]
                self.non_important_species = (input_data.readline()).split(' ')[:10]  # unimportant-species
                self.non_important_species = [n.replace(',', '') for n in self.non_important_species]
            print(f"Successfully loaded Global configuration file.")
        except FileNotFoundError:
            print(f"Global configuration file NOT found under: \\source_files\\{global_configuration_file_name}")
            raise FileNotFoundError

        super().__init__(root_path, early_stop, test_GA, parallel, self.checkpoint_path, self.max_iteration)

        # global record
        self.IDT_error_for_detailed_mechanism = 1.0
        self.global_best_chrom = np.zeros(self.n_species)
        self.global_lowest_error = 100.0
        self.average_error_of_generations = []
        self.errors_of_generations = []

        # local record
        self.IDT_error = np.zeros((self.population,))
        self.normalized_size = np.ones((self.population,))

        # important_species_mask
        # Todo: remove hard-coding!
        try:
            temp_gas = GasForReduction(self.learnable_parameters, 0, self.non_important_species,
                                                self.previous_learnable_parameters, None, self.mechanism_yaml_path,
                                       important_species=self.important_species)
        except FileNotFoundError:
            raise FileNotFoundError

        self.important_species_encode = temp_gas.important_species_encode
        self.learnable_parameters.important_species_mask = self.important_species_encode

        self.remained_reactions_encode_through_sensitivity = np.ones(self.n_reactions, )
        self.remained_species_encode_through_sensitivity = np.ones(self.n_species, )
        self.non_important_species_encode = None

    def load_experience_data(self):
        '''
        load experienced data.
        '''
        # labels and learnable parameters
        self.idp_Label = IDT_Label(None, self.root_path, self.IDT_configuration_path, self.experimental_IDT_path)
        self.pfr_Label = PFR_Label(None, self.root_path)

    def define_chromosome(self):
        '''
        Define the optimization targets (reduction code here).
        '''
        self.learnable_parameters = ReductionCode(self.root_path, self.n_species, self.mutation_rate, self.population,
                                                  self.initial_population)

    def compute_error_for_one_gene(self, index):
        """ Perform evaluation without multi-processing.
        """
        queue = Queue()
        p = Process(target=compute_error_for_Reduction, args=(self, index, queue))
        p.start()
        p.join()
        index, error = queue.get()
        return error

    def ranking(self):
        """
        Perform evaluation with multi-processing.
        """
        start_time = time.time()
        if not self.parallel:
            errors = np.array([self.compute_error_for_one_gene(i) for i in tqdm(range(self.population))])
        else:
            queue = Queue()
            process_list = []
            total_process_count = 0
            for i in range(self.max_process_number):
                p = Process(target=compute_error_for_Reduction, args=(self, total_process_count, queue))
                total_process_count += 1
                p.start()
                process_list.append(p)

            index_and_errors = []
            for i in tqdm(range(self.population)):
                index_and_errors.append(queue.get())
                if total_process_count < self.population:
                    p = Process(target=compute_error_for_Reduction, args=(self, total_process_count, queue))
                    total_process_count += 1
                    p.start()
                    process_list.append(p)

            def sort_by_index(t):
                return t[0]

            sorted_index_and_errors = sorted(index_and_errors, key=sort_by_index)
            errors = []
            for i in range(self.population):
                errors.append(sorted_index_and_errors[i][1])
            errors = np.array(errors)

        # add size penalty
        for index in range(self.population):
            print(f"The {index}th chromosome has normalized IDP_error of {errors[index]}.")
            self.IDT_error[index] = errors[index]
            size_penalty = np.sum(self.learnable_parameters.chrom[index]) / self.n_species
            self.normalized_size[index] = size_penalty
            print(f"Size_penalty: {size_penalty}")
            errors[index] += size_penalty * 3.0

        # GA select the biggest one, but we want to minimize the error, so we put a negative here
        self.FitV = -errors
        end_time = time.time()
        print(f"Took time {end_time - start_time} seconds.")

    def save_parents(self):
        """
        Save the learnable parameters in self.previous_learnable_parameters.
        """
        previous_learnable_parameters = ReactionRateConstants(self.root_path, self.population,
                                                              self.learnable_parameters.chrom)
        self.previous_learnable_parameters = previous_learnable_parameters

    def runtime_record(self):
        """
        Save the chromosomes and fitness values after each iteration.
        """
        best_gene_index_of_generation = np.argmax(self.FitV)
        average_error_of_generation = -np.average(self.FitV)

        if -self.FitV[best_gene_index_of_generation] < self.global_lowest_error:
            self.global_lowest_error = -self.FitV[best_gene_index_of_generation]

        self.average_error_of_generations.append(average_error_of_generation)
        self.errors_of_generations.append(-self.FitV)

    def save_to_local(self, iteration_index):
        """
        Save the chromosomes and fitness values to checkpoint files.
        """
        temp_gas = GasForReduction(self.learnable_parameters, 0, self.non_important_species,
                                   self.previous_learnable_parameters, None, self.mechanism_yaml_path)
        columns = temp_gas.specie_names
        columns.append("Error")
        columns.append("Size")
        self.learnable_parameters.save_chrom2checkpoint(iteration_index, self.IDT_error, self.normalized_size, columns,
                                                        self.output_file_name, self.initial_iteration)

    def plot_results(self):
        """
        Visualize fitness values.
        """
        # Plt the average error and lowest error curve of this runtime.
        Y_history = pd.DataFrame(self.errors_of_generations)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.show()

        column1 = np.reshape(self.global_best_chrom, (-1, 1))
        matrix = np.hstack([column1, ])
        s1 = pd.DataFrame(matrix, columns=['code'])
        s1.to_excel('Best_result_reduction.xlsx', index=False, header=False)

    def sensitivity_analysis(self):
        """
        Compute and stores sensitivity of each reaction.
        Todo: remove hard-coding
        """
        # self.initialize_population()
        gas = GasForReduction(self.learnable_parameters, 0, self.non_important_species,
                              self.previous_learnable_parameters, None, self.mechanism_yaml_path)
        s = compute_sensitivity(self.idp_Label, gas.detailed_mechanism_gas)
        data1 = pd.DataFrame(s)
        data1.to_csv(self.root_path + f"\{self.sensitivities_file_name}")
        return 0

    def reduce_raw_mechanism_through_sensitivity(self):
        """
        Preprocessing by Reducing raw mechanism through sensitivity.
        Todo: remove hard-coding
        """
        # compute sensitivity series for all reactions (time consuming)
        # specie_sensitivities_wrt_OH = self.sensitivity_analysis()
        # read precomputed sensitivity series for all reactions from file
        try:
            specie_sensitivities_wrt_OH = pd.read_csv(self.root_path + f"\{self.sensitivities_file_name}").values[:, 1]
            print(f"Successfully loaded sensitivity file.")
        except FileNotFoundError:
            print(f"Sensitivity file NOT found under: \\source_files\\{self.sensitivities_file_name}")
            print("Start sensitivity analysis")
            self.sensitivity_analysis()
            raise FileNotFoundError

        # reduce reactions with low sensitivities
        try:
            raw_mechanism_gas = GasForReduction(self.learnable_parameters, 0, self.non_important_species,
                                                self.previous_learnable_parameters, None, self.mechanism_yaml_path)
        except FileNotFoundError:
            raise FileNotFoundError

        self.IDT_error_for_detailed_mechanism = compute_IDP_error(self.idp_Label,
                                                                  raw_mechanism_gas.detailed_mechanism_gas)
        print(f"IDT error for detailed mechanism: {self.IDT_error_for_detailed_mechanism}")
        self.non_important_species_encode = raw_mechanism_gas.non_important_species_encode

        reaction_pruning_rate = 0.1
        reaction_encode_cache = [np.ones((self.n_reactions,)).astype(int)]
        while True:
            least_k = np.argsort(specie_sensitivities_wrt_OH)[:int(self.n_reactions * reaction_pruning_rate)]
            inverse_reaction_encode = np.zeros((self.n_reactions,))
            for index in least_k:
                inverse_reaction_encode[index] = 1
            inverse_reaction_encode = np.dot(inverse_reaction_encode,
                                             raw_mechanism_gas.duplicate_matrix_for_detailed_mechanism)
            reaction_encode = np.where(inverse_reaction_encode == 0, 1, 0)

            IDP_error_for_pruned_mechanism = compute_IDP_error(self.idp_Label,
                                                               raw_mechanism_gas.get_mechanism_gas_with_reaction_encode(
                                                                   reaction_encode))
            print(
                f"IDT_error_for_pruned_mechanism={IDP_error_for_pruned_mechanism} for reaction_pruning_rate={reaction_pruning_rate}")
            if IDP_error_for_pruned_mechanism - self.IDT_error_for_detailed_mechanism > self.delta or reaction_pruning_rate >= 0.5:
                break
            reaction_encode_cache.append(reaction_encode)
            reaction_pruning_rate += 0.1

        remained_species_encode = np.dot(reaction_encode_cache[-1], raw_mechanism_gas.species_matrix)
        remained_species_encode = np.where(remained_species_encode > 0, 1, 0).astype(int)
        #  Todo: remove hard coding for inert gas
        remained_species_encode[:3] = 1
        print(
            f"End pruning, {np.sum(reaction_encode)} out of {self.n_reactions} reactions remained, {np.sum(remained_species_encode)} out of {self.n_species} species remained."
            f" (Inert gases are manually kept by hard coding.)")
        self.remained_reactions_encode_through_sensitivity = reaction_encode  # create mask for reactions that are deleted through sensitivity analysis
        self.remained_species_encode_through_sensitivity = remained_species_encode  # create mask for reactions that are deleted through sensitivity analysis
        self.learnable_parameters.remained_species_encode_through_sensitivity = remained_species_encode
        return raw_mechanism_gas

    def initialize_population_with_sensitivity(self):
        """
        Preprocessing by Reducing raw mechanism through sensitivity.
        Todo: remove hard-coding
        """
        raw_mechanism_gas = self.reduce_raw_mechanism_through_sensitivity()

        remained_non_important_species_encode = self.remained_species_encode_through_sensitivity & raw_mechanism_gas.non_important_species_encode
        print("Remained non important species:")
        for i in range(self.n_species):
            if remained_non_important_species_encode[i] == 1:
                print(f"{raw_mechanism_gas.specie_names[i]}")

        # drop 3 non-important species
        encode_for_non_important_species_permutation = np.array(
            binary_permutation(3, np.sum(remained_non_important_species_encode)))

        initial_population_for_pruned_reduction_method = np.zeros(
            (len(encode_for_non_important_species_permutation),
             self.n_species)) + self.remained_species_encode_through_sensitivity

        c = 0
        for i in range(self.n_species):
            if remained_non_important_species_encode[i] == 1:
                initial_population_for_pruned_reduction_method[:, i] -= encode_for_non_important_species_permutation[:,
                                                                        c]
                c += 1

        ########################################### save the initial generation as text
        save_as_text = []
        for encode in initial_population_for_pruned_reduction_method:
            t = ''
            for i in range(len(encode)):
                if encode[i] == 1:
                    t += raw_mechanism_gas.specie_names[i]
                    t += ','
            save_as_text.append(t)

        data1 = pd.DataFrame(save_as_text)
        data1.to_csv(self.root_path + f"\pruned_initial_population.csv")
        ########################################### end

        print(f"Number of possible initial chromosomes: {len(initial_population_for_pruned_reduction_method)}")
        print(f"Keep the first {self.population}.")

        random_index = np.random.choice(np.arange(len(initial_population_for_pruned_reduction_method)), self.population,
                                        replace=False)
        return initial_population_for_pruned_reduction_method[random_index]

    def initialize_population(self):
        """
        Initialize the optimization targets (reduction code here).
        """
        self.learnable_parameters.initialize_population(self.initialize_population_with_sensitivity())
        # self.learnable_parameters.initialize_population()

    def visualize(self):
        """
        Read checkpoint files and draw the accuracy curve and size curve.
        """
        min_errors = []
        all_errors = []
        min_sizes = []
        all_sizes = []
        i = 0
        while True:
            try:
                content = pd.read_csv(f"Checkpoints/{i}_{self.output_file_name}").to_numpy()
                min_errors.append(np.min(content[:, -2]))
                all_errors.append(content[:, -2])
                min_sizes.append(np.min(content[:, -1]))
                all_sizes.append(content[:, -1])
                i += 1
            except FileNotFoundError:
                break

        if len(min_errors) == 0:
            raise FileNotFoundError

        # Plt the average error and lowest error curve of this runtime.
        fig, ax = plt.subplots(2, 1)
        # ax[0].plot(np.arange(len(min_errors)), min_errors, '.', color='red')
        for iteration_index in range(len(all_errors)):
            ax[0].plot(np.ones(len(all_errors[iteration_index])) * iteration_index, np.log10(all_errors[iteration_index]), '.', markersize=0.8, color='red')

        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('log-normalized-disagreement')
        # ax[1].plot(np.arange(len(min_sizes)), min_sizes, '.', color='b')
        for iteration_index in range(len(all_errors)):
            ax[1].plot(np.ones(len(all_sizes[iteration_index])) * iteration_index, all_sizes[iteration_index], '.', markersize=0.8, color='red')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Percentage of remaining species')
        plt.show()

    def makeGrid(self):
        """
        Read checkpoint files and create the grid images and a GIF that visualizes the populations.
        """
        min_errors = []
        min_sizes = []
        grid_file_name = []
        index = 0
        while True:
            try:
                content = pd.read_csv(f"Checkpoints/{index}_{self.output_file_name}").to_numpy()
                min_errors.append(np.min(content[:, -2]))
                min_sizes.append(np.min(content[:, -1]))

                genes = content[:, 1:-2]
                genes = np.array(genes).T
                # Create a color map using the 'jet' colormap
                # Create a figure and axis object
                fig, ax = plt.subplots()
                # Create a grid of squares and color them based on the grid values
                for i in range(genes.shape[0]):
                    for j in range(genes.shape[1]):
                        c = genes[i, j] * 0.75 + 0.25
                        ax.add_patch(plt.Rectangle((i, j), 1, 1, linewidth=0.2, edgecolor='black', facecolor=(c, c, c)))

                # Set the limits of the axis
                ax.set_xlim([0, genes.shape[0]])
                ax.set_ylim([0, genes.shape[1]])
                plt.savefig(f'grids/{self.output_file_name}_{index}.png')
                grid_file_name.append(f'grids/{self.output_file_name}_{index}.png')
                print(f"One image saved to [grids/{self.output_file_name}_{index}.png].")
                index += 1
            except FileNotFoundError:
                break

        if len(min_errors) == 0:
            raise FileNotFoundError

        images = [imageio.imread(filename) for filename in grid_file_name]
        # Save the images as a GIF
        imageio.mimsave('grids/animation.gif', images, duration=0.3)
        print(f"One GIF saved to [grids/animation.gif]")

    def generate_yaml_file(self, checkpoint_path_for_output, index, prefix):
        """
        Create a Cantera file for the mechanism described by the [index]-th chromosome in the current population.
        """
        """
        Create a Cantera file for the mechanism described by the [index]-th chromosome in the current population.
        """
        try:
            def read_initial_population(file_name):
                chrom = pd.read_csv(f"Checkpoints\\{file_name}")
                return chrom.values[:, 1:]

            output_population = read_initial_population(checkpoint_path_for_output)
            print(f"Checkpoint file loaded from path [Checkpoints\\{checkpoint_path_for_output}].")
        except FileNotFoundError:
            print(f"No checkpoint file loaded from path [Checkpoints\\{checkpoint_path_for_output}].")
            raise FileNotFoundError

        assert len(output_population) == self.population, "Please set population equal to number of individuals in the checkpoint file."
        output_learnable_parameters = ReductionCode(self.root_path, self.n_species, self.mutation_rate, self.population,
                      initial_population = output_population)
        output_learnable_parameters.initialize_population()

        if index is None:
            for i in range(self.population):
                gas = GasForReduction(output_learnable_parameters, i, self.non_important_species,
                                      self.previous_learnable_parameters, self.remained_reactions_encode_through_sensitivity,
                                      self.mechanism_yaml_path)
                string = gas.get_skeleton_mechanism_yaml_string(with_all_species=False)
                output_file_name = f"output\\{prefix}_individual{i}.yaml"
                f = open(output_file_name, mode="w")
                print(f"Reduced mechanism saved to [{output_file_name}].\n")
                f.write(string)
                f.close()

        else:
            gas = GasForReduction(output_learnable_parameters, index, self.non_important_species,
                                  self.previous_learnable_parameters, self.remained_reactions_encode_through_sensitivity,
                                  self.mechanism_yaml_path)
            string = gas.get_skeleton_mechanism_yaml_string(with_all_species=False)
            output_file_name = f"output\\{prefix}_individual{index}.yaml"
            f = open(output_file_name, mode="w")
            print(f"Reduced mechanism saved to [{output_file_name}].\n")
            f.write(string)
            f.close()


    def print_IDT_results(self, index):
        ct.suppress_thermo_warnings()
        self.learnable_parameters.chrom = self.initial_population[:, :-2]
        gas = GasForReduction(self.learnable_parameters, index, self.non_important_species,
                              self.previous_learnable_parameters, self.remained_reactions_encode_through_sensitivity,
                              self.mechanism_yaml_path)
        IDP_error = compute_IDP_error(self.idp_Label, gas.get_skeleton_mechanism_gas(),
                                      average_rate=0.5, save_path="output/IDT.csv", save=True)




def compute_error_for_Reduction(ga: GeneticAlgorithmForReduction, index, queue):
    """ Compute the error that consists only of IDP_error.
    """
    if ga.test_GA:
        error = np.sum(ga.learnable_parameters.chrom[index])
    else:
        ct.suppress_thermo_warnings()
        gas = GasForReduction(ga.learnable_parameters, index, ga.non_important_species,
                              ga.previous_learnable_parameters, ga.remained_reactions_encode_through_sensitivity,
                              ga.mechanism_yaml_path)
        IDP_error = compute_IDP_error(ga.idp_Label, gas.get_skeleton_mechanism_gas(),
                                      average_rate=0.5) / ga.IDT_error_for_detailed_mechanism
        error = IDP_error
    queue.put((index, error))
