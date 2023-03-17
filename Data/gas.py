# from abc import abstractmethod

import numpy as np
import cantera as ct
from cantera import CanteraError
import re
from Data.learnable_parameters import ReactionRateConstants, ReductionCode, LearnableParameter
from Tools.tools import equation_equal


class Gas:
    '''
    This is an interface to Cantera, which produces the mechanism object for simulation and evaluation
    '''
    def __init__(self, learnableParameters: LearnableParameter, index: int, previousLearnableParameters, mechanism_yaml_path="TUM_CH2O_CH3OH_0.2.yaml"):
        '''
        The Cantera fuel object is produced by a description text, namely a long String.
        Gas class reads, decomposes, and stores the description text.
        A LearnableParameter object is saved as a parameter, which provides information to manipulate the description text to operate on the mechanism indirectly.
        The kernel function get_gas() returns a Cantera object for simulation.

        :param learnableParameters: values for optimization targets needed for description text manipulation.
        :param index: the index of the produced mechanism in the population (defined in learnableParameters).
        :return: None
        '''
        self.root_path = learnableParameters.root_path
        self.learnableParameters = learnableParameters
        self.previousLearnableParameters = previousLearnableParameters
        self.gene_index = index

        try:
            # read '.yaml' file template for Cantera use
            with open(self.root_path + "\source_files" + f"\{mechanism_yaml_path}", 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # print(f"Detailed mechanism file found from path [source_files\\{mechanism_yaml_path}].")
            # print(f"Processing detailed mechanism file...")
        except FileNotFoundError:
            print(f"No detailed mechanism found under path [source_files\\{mechanism_yaml_path}].")
            raise FileNotFoundError

        try:
            self.reactions_texts = []
            reaction_text = ''
            for line in lines:
                if not line.startswith('- equation:'):
                    reaction_text += line
                else:
                    self.reactions_texts.append(reaction_text)
                    reaction_text = line
            self.reactions_texts.append(reaction_text)

            # split header
            self.header = self.reactions_texts[0]
            self.reactions_texts = self.reactions_texts[1:]

            # save detailed mechanism
            description = '' + self.header
            for reaction_text in self.reactions_texts:
                description += reaction_text
            self.detailed_mechanism_gas = ct.Solution(yaml=description)
            self.n_reactions = len(self.detailed_mechanism_gas.reactions())

            reactions = self.detailed_mechanism_gas.reactions()
            self.duplicate_matrix_for_detailed_mechanism = np.zeros((self.n_reactions, self.n_reactions))

            for i in range(self.n_reactions):
                for j in range(self.n_reactions):
                    if equation_equal(reactions[i].equation, reactions[j].equation):
                        self.duplicate_matrix_for_detailed_mechanism[i, j] = 1
            # print(f"Detailed mechanism processing succeed...")
        except RuntimeError:
            print(f"Detailed mechanism processing failed, please check file format.")
            raise FileNotFoundError


class GasForOptimization(Gas):
    '''
    This is an interface to Cantera, which produces the mechanism object for RRC optimization tasks.
    '''

    def __init__(self, learnableParameters: ReactionRateConstants, index: int, previousLearnableParameters,
                 mechanism_yaml_path="TUM_CH2O_CH3OH_0.2.yaml", optimization_pointers_file="mech_CH2O_CH3OH.dat"):
        super().__init__(learnableParameters, index, previousLearnableParameters, mechanism_yaml_path)
        '''
        The Cantera fuel object is produced by a description text, namely a long String.
        Gas class reads, decomposes, and stores the description text.
        A LearnableParameter object is saved as a parameter, which provides information to manipulate the description text to operate on the mechanism indirectly.
        The kernel function get_gas() returns a Cantera object for simulation.
        
        :param learnableParameters: values for optimization targets needed for description text manipulation.
        :param index: the index of the produced mechanism in the population (defined in learnableParameters).
        :param mechanism_yaml_path: path for the input detailed mechanism.
        :param optimization_pointers_file: path for the file that locates RRCs to optimize).
        :return: None
        '''

        # read learnable parameter locations on the template
        self.locations = []
        self.original_text = []
        try:
            with open(self.root_path + "\source_files" + f"\{optimization_pointers_file}", 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[5:]:
                    pieces = line.split('#')
                    if len(pieces) > 1:
                        location = pieces[-1].split(' ')[-1]
                        self.locations.append(location)
                        self.original_text.append(pieces[0])
            # print(f"RRC location file found under path [source_files\\{optimization_pointers_file}].")
            # print("Processing RRC locations file.")
        except FileNotFoundError:
            print(f"RRC location file NOT found under path [source_files\\{optimization_pointers_file}].")
            raise FileNotFoundError
        except RuntimeError:
            print(f"RRC location file processing failed, please check file format.")
            raise FileNotFoundError

    def get_gas(self, return_description=False):
        '''
        Return a Cantera object for simulation, which is described by the "index"-th chromosome in self.learnableParameters.
        '''
        # turn X into real values
        assert isinstance(self.learnableParameters,
                          ReactionRateConstants), "Learnable parameter during optimization must be RRCs"
        new_parameter_values = self.learnableParameters.get_real_values()[self.gene_index]
        RRC_indexer = self.learnableParameters.get_RRC_indexer()

        # read learnable parameter values to fit into locations above
        specified_reaction_texts = self.reactions_texts.copy()
        for i, location in enumerate(self.locations):
            if location.endswith('\n'):
                location = location[:-1]

            if location.endswith('L'):
                reaction_number = int(location[:-1])
                index_in_X = RRC_indexer['NORMAL'].pop()
                new_text = str(str('A: ' + str('{:.6e}'.format(10 ** new_parameter_values[index_in_X]))))
                specified_reaction_texts[reaction_number - 1] = self.reactions_texts[reaction_number - 1].replace(
                    self.original_text[i], new_text)

            elif location.endswith('H'):
                reaction_number = int(location[:-1])
                index_in_X = RRC_indexer['H'].pop()
                new_text = str('A: ' + str('{:.6e}'.format(10 ** new_parameter_values[index_in_X])))
                specified_reaction_texts[reaction_number - 1] = self.reactions_texts[reaction_number - 1].replace(
                    self.original_text[i], new_text)
            elif location.endswith('N2'):
                reaction_number = int(location[:-2])
                index_in_X = RRC_indexer['N2'].pop()
                new_text = str('N2: ' + str(new_parameter_values[index_in_X]))
                specified_reaction_texts[reaction_number - 1] = self.reactions_texts[reaction_number - 1].replace(
                    self.original_text[i], new_text)
            elif location.endswith('AR'):
                reaction_number = int(location[:-2])
                index_in_X = RRC_indexer['AR'].pop()
                new_text = str('AR: ' + str(new_parameter_values[index_in_X]))
                specified_reaction_texts[reaction_number - 1] = self.reactions_texts[reaction_number - 1].replace(
                    self.original_text[i], new_text)
            else:
                reaction_number = int(location)
                index_in_X = RRC_indexer['NORMAL'].pop()
                new_text = str('A: ' + str('{:.6e}'.format(10 ** new_parameter_values[index_in_X])))
                specified_reaction_texts[reaction_number - 1] = self.reactions_texts[reaction_number - 1].replace(
                    self.original_text[i], new_text)

        # save original gas
        description = '' + self.header
        for reaction_text in specified_reaction_texts:
            description += reaction_text

        if return_description:
            return description

        try:
            g = ct.Solution(yaml=description)
            return g
        except CanteraError:
            random_index = np.random.randint(0, self.learnableParameters.population)
            if self.previousLearnableParameters is None:
                print(
                    "This is a CanteraError! Reaction rate parameters not acceptable for fuel initialization. This gene will be replaced by a new random one.")
                newLearnableParameters = ReactionRateConstants(self.root_path, self.learnableParameters.population, None)
                g = GasForOptimization(newLearnableParameters, random_index, None)
            else:
                print(
                    "This is a CanteraError! Reaction rate parameters not acceptable for fuel initialization. This gene will be replaced by a random one from previous generation.")
                g = GasForOptimization(self.previousLearnableParameters, random_index, None)
            return g
        except RuntimeError:
            print(
                "This is not a known CanteraError! It is a RuntimeError happened during fuel initialization. Please check what has happened!")
            exit()


class GasForReduction(Gas):
    '''
    This is an interface to Cantera, which produces the mechanism object for reduction tasks.
    '''
    def __init__(self, learnableParameters: ReductionCode, index: int, non_important_species, previousLearnableParameters
                 , remained_reactions_encode_through_sensitivity=None, mechanism_yaml_path="TUM_CH2O_CH3OH_0.2.yaml"):
        super().__init__(learnableParameters, index, previousLearnableParameters, mechanism_yaml_path)
        '''
        The Cantera fuel object is produced by a description text, namely a long String.
        Gas class reads, decomposes, and stores the description text.
        A LearnableParameter object is saved as a parameter, which provides information to manipulate the description text to operate on the mechanism indirectly.
        The kernel function get_gas() returns a Cantera object for simulation.
        
        :param learnableParameters: values for optimization targets needed for description text manipulation.
        :param index: the index of the produced mechanism in the population (defined in learnableParameters).
        :param previousLearnableParameters: learnableParameters in the previous iteration.
        :param remained_reactions_encode_through_sensitivity: as name described.
        :param mechanism_yaml_path: path for the input detailed mechanism.
        :return: None
        '''
        if remained_reactions_encode_through_sensitivity is None:
            self.remained_reactions_encode_through_sensitivity = np.ones((self.n_reactions,)).astype(int)
        else:
            self.remained_reactions_encode_through_sensitivity = remained_reactions_encode_through_sensitivity.astype(int)
        self.n_reactions = len(self.detailed_mechanism_gas.reactions())
        self.n_species = len(self.detailed_mechanism_gas.species_names)
        self.species_matrix = self.get_species_matrix()

        if self.learnableParameters.chrom is None:
            # print("GasForReduction Class receives empty input chromosome, initializing with all detailed mechanisms.")
            self.chrom_specie_encode = np.ones((self.learnableParameters.population, self.n_species))
        else:
            self.chrom_specie_encode = self.learnableParameters.chrom

        self.skeleton_mechanisms = np.dot(self.species_matrix, (1 - self.chrom_specie_encode).T).T
        self.skeleton_mechanisms = np.where(self.skeleton_mechanisms == 0, 1, 0)

        # Todo: mask all reactions that have been cropped through sensitivity analysis
        for i in range(len(self.skeleton_mechanisms)):
            self.skeleton_mechanisms[i] = self.remained_reactions_encode_through_sensitivity & self.skeleton_mechanisms[i]

        self.header_before_phase_description, self.specie_names, self.header_before_species_after_phase, self.header_species, self.header_end = self.split_header()

        self.non_important_species_encode = np.zeros((self.n_species,)).astype(int)
        for i in range(self.n_species):
            if self.specie_names[i] in non_important_species:
                self.non_important_species_encode[i] = 1


    def get_skeleton_mechanism_gas(self, with_all_species=True):
        '''
        Return a Cantera object for simulation, which is described by the "index"-th chromosome in self.learnableParameters.
        '''

        # Todo: problem occurs then deleted specie has a third-body efficiencies in some reactions
        code = self.chrom_specie_encode[self.gene_index]
        if with_all_species:
            code = np.ones((self.n_species,))
        # Todo: problem end

        description = '' + self.header_before_phase_description
        description += "species: ["

        remaining_species_names = [self.specie_names[i] for i in np.where(code == 1)[0]]

        for name in remaining_species_names:
            description += name + ','
        description = description[:-1]
        description += "]" + self.header_before_species_after_phase + "species:\n"
        for i in range(self.n_species):
            if code[i] == 1:
                description += self.header_species[i]
        description += self.header_end
        for i, reaction_text in enumerate(self.reactions_texts):
            if self.skeleton_mechanisms[self.gene_index, i] == 1:
                description += reaction_text
        r = ct.Solution(yaml=description)
        return r


    def get_skeleton_mechanism_yaml_string(self, with_all_species=True):
        '''
        Return the description text for the Cantera object, which is described by the "index"-th chromosome in self.learnableParameters.
        '''

        # Todo: problem occurs then deleted specie has a third-body efficiencies in some reactions
        code = self.chrom_specie_encode[self.gene_index]
        if with_all_species:
            code = np.ones((self.n_species,))
        # Todo: problem end

        description = '' + self.header_before_phase_description
        description += "species: ["

        remaining_species_names = [self.specie_names[i] for i in np.where(code == 1)[0]]

        for name in remaining_species_names:
            description += name + ','
        description = description[:-1]
        description += "]" + self.header_before_species_after_phase + "species:\n"
        for i in range(self.n_species):
            if code[i] == 1:
                description += self.header_species[i]
        description += self.header_end
        for i, reaction_text in enumerate(self.reactions_texts):
            if self.skeleton_mechanisms[self.gene_index, i] == 1:
                description += reaction_text
        return description

    def get_mechanism_gas_with_reaction_encode(self, reaction_encode):
        '''
        Helping method that decomposes the description text.
        '''
        specie_code = np.ones((self.n_species,))
        description = '' + self.header_before_phase_description
        description += "species: ["

        remaining_species_names = [self.specie_names[i] for i in np.where(specie_code == 1)[0]]
        for name in remaining_species_names:
            description += name + ','
        description = description[:-1]
        description += "]" + self.header_before_species_after_phase + "species:\n"
        for i in range(self.n_species):
            if specie_code[i] == 1:
                description += self.header_species[i]
        description += self.header_end
        for i, reaction_text in enumerate(self.reactions_texts):
            if reaction_encode[i] == 1:
                description += reaction_text
        return ct.Solution(yaml=description)

    def get_species_matrix(self):
        '''
        Return a matrix that describes which species participates in which reactions.
        '''
        species_matrix = np.zeros((self.n_reactions, self.n_species))

        def get_reactants_and_products(reaction):
            return list(reaction.products.keys()) + list(reaction.reactants.keys())

        for reaction_index in range(self.n_reactions):
            reactants_and_products = get_reactants_and_products(self.detailed_mechanism_gas.reaction(reaction_index))
            for specie_index in range(self.n_species):
                if self.detailed_mechanism_gas.species_names[specie_index] in reactants_and_products:
                    species_matrix[reaction_index, specie_index] = 1
        return species_matrix

    def split_header(self):
        '''
        Helping method that decomposes the description text.
        '''

        split = self.header.split("species: [")
        header_before_phase_description = split[0]
        split = split[1]
        split = split.split("]", maxsplit=1)
        phase_summary_text = split[0]
        phase_summary_text = re.sub("\n", "", phase_summary_text, 0)
        phase_summary_text = re.sub(" ", "", phase_summary_text, 0)
        specie_names = phase_summary_text.split(",")

        header_after_phase = split[1]

        split = header_after_phase.split("species:", maxsplit=1)
        header_before_species_after_phase = split[0]
        header_species = split[1]

        splitted_header_species = header_species.split("- name: ")

        species_texts = ["- name: " + s for s in splitted_header_species[1:]]
        header_end = species_texts[-1][-12:]  # "-reactions:"
        species_texts[-1] = species_texts[-1][:-12]
        return header_before_phase_description, specie_names, header_before_species_after_phase, species_texts, header_end
