import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from abc import abstractmethod, ABCMeta


class LearnableParameter(metaclass=ABCMeta):
    """This abstract class defines the behavior of optimization targets and its corresponding encodings abstractly.
    """

    def __init__(self, root_path, population, initial_population):
        """This abstract class defines the behavior of optimization targets and its corresponding encodings.

        Parameters
        ----------
        root_path: String
            Root path of the program.
        population: int
            Population size.
        initial_population: 2D np array
            Inputted initial population if exists.
        Returns
        -------
        List[float]:
            Reactions sensitivities.
        """
        self.name = "LearnableParametersInterface"
        self.root_path = root_path
        self.population = population
        self.initial_population = initial_population
        self.chrom = None

    @abstractmethod
    def save_chrom2checkpoint(self, iteration):
        raise NotImplementedError

    @abstractmethod
    def mutation(self):
        raise NotImplementedError

    @abstractmethod
    def crossover(self):
        raise NotImplementedError

    @abstractmethod
    def selection(self, FitV):
        raise NotImplementedError

    @abstractmethod
    def initialize_population(self):
        raise NotImplementedError

    def selection_tournament_faster(self, FitV, tourn_size=4):
        '''
        Select the best individual among *tournsize* randomly chosen
        Same with `selection_tournament` but much faster using numpy
        individuals,
        :param FitV: Fitness value
        :param tourn_size: size of a tournament
        :return: None
        '''
        # create tournaments
        aspirants_idx = np.random.randint(self.population, size=(self.population, tourn_size))
        aspirants_values = FitV[aspirants_idx]
        # select parents by performing tournaments
        winner = aspirants_values.argmax(axis=1)  # winner index in every team
        sel_index = [aspirants_idx[i, j] for i, j in enumerate(winner)]
        self.chrom = self.chrom[sel_index, :]


class ReactionRateConstants(LearnableParameter):
    """This class loads, saves, encodes ReactionRateConstants.

    RRC <==> X <==> normalized X <==> Chrom
            |---------- chrom2x ----------|

    """

    def __init__(self, root_path, population, initial_population):
        """Read the input files for the detailed mechanism and RRCs to optimize.

        Parameters
        ----------
        root_path: String
            Root path of the program.
        population: int
            Population size.
        initial_population: 2D np array
            Inputted initial population if exists.
        """
        super().__init__(root_path, population, initial_population)
        self.description = 'Optimization goals are "Fitting Factors" and will be referred as "X" in the code. '

        # read upper and lower bound for X
        # Todo: clear redundant data
        with open(root_path + "\source_files" + "\GAinputOptimization.dat", 'r') as input_data:
            self.dimension = int((input_data.readline()).split(' ', 1)[0])  # Dimension
            self.mutation_rate = float((input_data.readline()).split(' ', 1)[0])  # Mutation rate
            self.population = int((input_data.readline()).split(' ', 1)[0])  # Population size
            self.max_iteration = int((input_data.readline()).split(' ', 1)[0])  # Maximum iteration
            initial_iteration = int((input_data.readline()).split(' ', 1)[0])  # initial iteration
            precision = float((input_data.readline()).split(' ', 1)[0])  # precision
            self.precision = np.array(precision) * np.ones(self.dimension)
            self.X_upper_bound = np.zeros(self.dimension)  # X upper bond
            self.X_lower_bound = np.zeros(self.dimension)  # X lower bond
            max_process = int((input_data.readline()).split(' ', 1)[0])  # Maximum process
            checkpoint = (input_data.readline()).split(' ', 1)[0]  # checkpoint
            mechanism_yaml_path = (input_data.readline()).split(' ', 1)[0]  # mechanism yaml path
            IDT_configuration_path = (input_data.readline()).split(' ', 1)[0]  # IDT configuration path
            experimental_IDT_path = (input_data.readline()).split(' ', 1)[0]  # experimental IDT path
            PFR_configuration_path = (input_data.readline()).split(' ', 1)[0]  # PFR configuration path
            experimental_PFR_path = (input_data.readline()).split(' ', 1)[0]  # experimental PFR path
            optimization_pointers_file = (input_data.readline()).split(' ', 1)[0]  # optimization_pointers_file
            self.optimization_interval_file = (input_data.readline()).split(' ', 1)[0]  # optimization interval
            output_file_name = (input_data.readline()).split(' ', 1)[0]  # output_file_name

            # 遍历input.dat文件中剩下的数据，同样也要对字符串形式的数据进行处理分离
            # 第一次循环以前，读取位置为第二行最开始
            for i in range(self.dimension):  # 读取变量上下限
                line = (input_data.readline()).split(' ', 2)
                self.X_lower_bound[i] = float(line[0])  # 每一行第一个数为变量下限值
                self.X_upper_bound[i] = float(line[1])  # 每一行第二个数为变量上限值
        # end

        # read upper and lower bound for RRC
        try:
            with open(root_path + "\source_files" + f"\{self.optimization_interval_file}", 'r') as input_data:
                self.RRCNum_total = int((input_data.readline()).split(' ', 1)[0])  # RRC个数
                self.RRCNum = int((input_data.readline()).split(' ', 1)[0])  # RRC个数
                self.RRCEFNum = int((input_data.readline()).split(' ', 1)[0])  # Third-body efficy 个数
                self.RRC_N2_indexes = input_data.readline().split('#', 1)[0]  # indexes of learnable parameters for N2
                self.RRC_N2_indexes = self.RRC_N2_indexes.split()
                self.RRC_N2_indexes = list(int(i) for i in self.RRC_N2_indexes)
                self.RRC_Ar_indexes = input_data.readline().split('#', 1)[0]  # indexes of learnable parameters for AR
                self.RRC_Ar_indexes = self.RRC_Ar_indexes.split()
                self.RRC_Ar_indexes = list(int(i) for i in self.RRC_Ar_indexes)
                self.RRC_high_pressure_indexes = input_data.readline().split('#', 1)[
                    0]  # indexes of learnable parameters for RRC High pressure RRC
                self.RRC_high_pressure_indexes = self.RRC_high_pressure_indexes.split()
                self.RRC_high_pressure_indexes = list(int(i) for i in self.RRC_high_pressure_indexes)

                self.RRC_upper_bound = np.zeros(self.RRCNum_total)  # RRC 上限
                self.RRC_lower_bound = np.zeros(self.RRCNum_total)  # RRC 下限

                for i in range(self.RRCNum_total):  # 读取变量上下限
                    line = (input_data.readline()).split(' ', 2)
                    self.RRC_lower_bound[i] = float(line[0])  # 每一行第一个数为变量下限值
                    self.RRC_upper_bound[i] = float(line[1])  # 每一行第二个数为变量上限值
            print(f"Upper and lower bound for RRCs loaded from path [source_files\\{self.optimization_interval_file}].")
        except FileNotFoundError:
            print(
                f"No upper and lower bound for RRCs found under path [source_files\\{self.optimization_interval_file}].")
            raise FileNotFoundError
        except ValueError:
            print(
                f"ValueError occurred reading upper and lower bound for RRCs from [source_files\\{self.optimization_interval_file}].")
            raise FileNotFoundError

        # Todo: remove redundant input hyper-parameter
        self.population = population  # set population manually

        # construct X and chrom
        # Lind is the num of genes of every variable of func(segments)
        Lind_raw = np.log2((self.X_upper_bound - self.X_lower_bound) / self.precision + 1)
        self.Lind = np.ceil(Lind_raw).astype(int)
        self.len_chrom = sum(self.Lind)
        self.X = None
        # self.initialize_population()

    def initialize_population(self):
        """Initialize with given first population or otherwise the randomly initialized population.
        """
        # initialize population
        if self.initial_population is None:
            self.chrom = np.random.randint(low=0, high=2, size=(int(self.population), int(self.len_chrom)))
        else:
            assert (np.shape(self.initial_population)[0] == self.population), \
                f"Initial population (of size {np.shape(self.initial_population)[0]}) does not satisfy population configuration ({self.population})."
            assert (np.shape(self.initial_population)[1] == self.len_chrom + 1), \
                f"Length of initial genes ({np.shape(self.initial_population)[1]}) does not satisfy configuration ({self.len_chrom})."
            self.chrom = self.initial_population[:, :self.len_chrom]
        self.X = self.chrom2x()

    def selection(self, FitV):
        """Perform tournament selection.
        """
        self.selection_tournament_faster(FitV, tourn_size=3)
        self.chrom2x()

    def save_chrom2checkpoint(self, iteration, initial_iteration=0, output_name="Optimization", fitness=None):
        shape = np.shape(self.chrom)
        data = np.zeros((shape[0], shape[1] + 1))
        data[:, :shape[1]] = self.chrom

        if not fitness is None:
            data[:, -1] = fitness

        data1 = pd.DataFrame(data)
        data1.to_csv(self.root_path + "\Checkpoints" + f"\{iteration + initial_iteration}_{output_name}")

    @staticmethod
    def gray2rv(gray_code):
        """Gray Code to real value: one piece of a whole chromosome
        input is a 2-dimensional numpy array of 0 and 1.
        output is a 1-dimensional numpy array which convert every row of input into a real number.
        """
        # Gray Code to real value: one piece of a whole chromosome
        # input is a 2-dimensional numpy array of 0 and 1.
        # output is a 1-dimensional numpy array which convert every row of input into a real number.
        _, len_gray_code = gray_code.shape
        b = gray_code.cumsum(axis=1) % 2  # transform to binary code
        mask = np.logspace(start=1, stop=len_gray_code, base=0.5, num=len_gray_code)
        return (b * mask).sum(axis=1) / mask.sum()  # transform to real value (normalized X)

    def chrom2x(self):
        """Transform binary chromosome to "real valued chromosome" X.
        """
        cumsum_len_segment = self.Lind.cumsum()
        X = np.zeros(shape=(self.population, self.dimension))
        for i, j in enumerate(cumsum_len_segment):
            if i == 0:
                Chrom_temp = self.chrom[:, :cumsum_len_segment[0]]
            else:
                Chrom_temp = self.chrom[:, cumsum_len_segment[i - 1]:cumsum_len_segment[i]]
            X[:, i] = self.gray2rv(Chrom_temp)  # get normalized X

        self.X = self.X_lower_bound + (self.X_upper_bound - self.X_lower_bound) * X  # de-normalizing
        return self.X

    def mutation(self):
        """
        Source: SKO.mutation.mutation
        mutation of 0/1 type chromosome
        faster than `self.Chrom = (mask + self.Chrom) % 2`
        """
        shape = np.shape(self.chrom)
        mask = (np.random.rand(*shape) < self.mutation_rate)
        self.chrom ^= mask
        self.chrom2x()

    def crossover(self):
        """
        Source: SKO.crossover.crossover_2point_bit
        3 times faster than `crossover_2point`, but only use for 0/1 type of Chrom
        """
        Chrom, size_pop, len_chrom = self.chrom, self.population, self.len_chrom
        half_size_pop = int(size_pop / 2)
        Chrom1, Chrom2 = Chrom[:half_size_pop], Chrom[half_size_pop:]
        mask = np.zeros(shape=(half_size_pop, len_chrom), dtype=int)
        # mask determines which part of gene will be exchanged
        for i in range(half_size_pop):
            n1, n2 = np.random.randint(0, self.len_chrom, 2)
            if n1 > n2:
                n1, n2 = n2, n1
            mask[i, n1:n2] = 1
        # 'invert a bit if mask==1 and parents are different' is equivalent to 'exchange a bit if mask==1'
        mask2 = (Chrom1 ^ Chrom2) & mask
        Chrom1 ^= mask2
        Chrom2 ^= mask2
        self.chrom2x()

    def get_real_values(self):
        """
        Denormalize X.
        """
        real_values = self.RRC_lower_bound + self.X * (self.RRC_upper_bound - self.RRC_lower_bound)
        return real_values

    def get_RRC_indexer(self):
        """
        Locate RRCs in the mechanism text description.
        Generalization needed.
        """
        abnormal_indexes = (self.RRC_N2_indexes + self.RRC_Ar_indexes + self.RRC_high_pressure_indexes)
        abnormal_indexes.sort()
        normal_index = [i for i in range(self.RRCNum_total) if (i not in abnormal_indexes)]

        normal_index.reverse()
        N2 = self.RRC_N2_indexes.copy()
        N2.reverse()
        Ar = self.RRC_Ar_indexes.copy()
        Ar.reverse()
        hp = self.RRC_high_pressure_indexes.copy()
        hp.reverse()
        RRC_indexer = {'NORMAL': normal_index,
                       'N2': N2,
                       'AR': Ar,
                       'H': hp}
        return RRC_indexer

    def show_bounds(self):
        """
        Visualize RRCs.
        """
        x = range(self.dimension)
        plt.plot(x, self.X_lower_bound, marker='o', mec='r', mfc='w', label='lower_bound')
        plt.plot(x, self.X_upper_bound, marker='*', ms=10, label='upper_bound')
        plt.legend()
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel('learnable parameters')
        plt.ylabel("value")
        plt.title("Parameter bounds")
        plt.show()

    def show_x(self, index):
        """
        Visualize X.
        """
        x = range(self.dimension)
        plt.plot(x, self.X_lower_bound, marker='o', mec='r', mfc='w', label='lower_bound')
        plt.plot(x, self.X_upper_bound, marker='*', ms=10, label='upper_bound')
        plt.plot(x, self.X[index], marker='*', ms=10, label='X')
        plt.legend()
        plt.margins(0)
        plt.subplots_adjust(bottom=0.15)
        plt.xlabel('learnable parameters')
        plt.ylabel("value")
        plt.title("Parameter bounds")
        plt.show()


class ReductionCode(LearnableParameter):
    """This class encodes and saves the reduced mechanisms.
    """

    def __init__(self, root_path, n_species, mutation_rate, population, initial_population,
                 remained_species_encode_through_sensitivity=None):
        """Initialize population and configure hyper parameters.
        """

        super().__init__(root_path, population, initial_population)
        # Todo: keep hyper-parameters into a dictionary
        self.n_species = n_species
        self.mutation_rate = mutation_rate
        self.important_species_mask = np.zeros((self.population, self.n_species))  # important species are masked by 1
        if remained_species_encode_through_sensitivity is None:
            self.remained_species_encode_through_sensitivity = np.ones((self.n_species,)).astype(int)
        else:
            self.remained_species_encode_through_sensitivity = remained_species_encode_through_sensitivity.astype(int)

    def initialize_population(self, initial_population_produced_by_sensitivity=None):
        """Initialize population with sensitivity analysis and otherwise with all detailed mechanisms.
        """

        if self.initial_population is None:
            if initial_population_produced_by_sensitivity is None:
                # simple initialization
                self.chrom = np.ones((int(self.population), int(self.n_species)))
            else:
                # initialization with sensitivity analysis
                self.chrom = initial_population_produced_by_sensitivity
        else:
            assert (np.shape(self.initial_population)[0] == self.population), \
                f"Initial population (of size {np.shape(self.initial_population)[0]}) does not satisfy population configuration ({self.population})."
            # assert (np.shape(self.initial_population)[1] == self.n_species), \
            #     f"Length of initial genes ({np.shape(self.initial_population)[1]}) does not satisfy configuration ({self.n_species})."
            self.chrom = self.initial_population[:, :self.n_species]
        self.chrom = self.chrom.astype(int) & self.remained_species_encode_through_sensitivity

    def selection(self, FitV):
        """
        Tournament selection.
        """
        self.selection_tournament_faster(FitV, tourn_size=3)

    def crossover(self):
        """
        Single-point crossover
        """
        Chrom, size_pop, len_chrom = self.chrom, self.population, self.n_species
        Chrom = Chrom.astype(int)
        half_size_pop = int(size_pop / 2)
        Chrom1, Chrom2 = Chrom[:half_size_pop], Chrom[half_size_pop:]
        mask = np.zeros(shape=(half_size_pop, len_chrom), dtype=int)
        # mask determines which part of gene will be exchanged
        for i in range(half_size_pop):
            n, = np.random.randint(0, self.n_species, 1)
            mask[i, :n] = 1
        # 'invert a bit if mask==1 and parents are different' is equivalent to 'exchange a bit if mask==1'
        mask = mask.astype(int)
        mask2 = (Chrom1 ^ Chrom2) & mask
        Chrom1 ^= mask2
        Chrom2 ^= mask2

    def mutation(self):
        """
        One-directional mutation: Every 1 has a possibility of [mutation_rate] to become 0 while maintaining important species
        """
        shape = np.shape(self.chrom)
        negative_mask = (np.random.rand(*shape) > self.mutation_rate)
        # maintain important species
        self.chrom = self.chrom.astype(int)
        self.chrom &= negative_mask
        self.chrom = self.chrom.astype(int)
        self.chrom |= self.important_species_mask.astype(int)

    def save_chrom2checkpoint(self, iteration, IDT_error=None, normalized_size=None, columns=None,
                              output_file_name=None, initial_iteration=None):
        """
        Store checkpoints.
        """
        if IDT_error is None:
            IDT_error = np.zeros((self.population,))
        if output_file_name is None:
            output_file_name = "Reduction.csv"
        if initial_iteration is None:
            initial_iteration = 0
        if normalized_size is None:
            normalized_size = np.zeros((self.population,))
        data = np.zeros((self.population, self.n_species + 2))
        data[:, :self.n_species] = self.chrom
        data[:, -1] = normalized_size
        data[:, -2] = IDT_error
        if columns is None:
            data1 = pd.DataFrame(data)
        else:
            data1 = pd.DataFrame(data, columns=columns)

        data1.to_csv(self.root_path + "\Checkpoints" + f"\{iteration + initial_iteration}_{output_file_name}")
