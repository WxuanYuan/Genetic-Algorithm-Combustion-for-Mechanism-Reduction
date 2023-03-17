# Genetic Algorithm Combustion Mechanism Reduction

This repository contains a Python program that uses a genetic algorithm to reduce the number of species in a combustion mechanism. This README file provides instructions on how to prepare the input files and how to start the program.

## Input Files

The following input files are required and should be placed in the `\source_files\` directory:

1. **General Configuration File**: This file sets all global hyperparameters and paths to all other input files. Provide the path to this file when launching the program.

   Example of a General Configuration File:

```
72 # Dimension: number of species in the input detailed mechanism
610 # Reaction: number of reactions in the input detailed mechanism
0.03 # Mutation rate
30 # Population size
30 # Maximum iteration: genetic algorithm will run for this number of iterations and terminates automatically
30 # Initial iteration: this number is only used to name checkpoints files
0.001 # Encoding precision: modification is not suggested
5 # Maximum process: number of processes for multi-processing, must be smaller than the number of CPU cores
TUM_CH2O_CH3OH_0.2.yaml # Mechanism yaml path
CH4_IDT_configuration.dat # IDT configuration path
CH4_experimental_IDT.csv # Experimental IDT path
59_CH4_Reduction_optimized_final.csv # Checkpoint path: the program will load this file and use it as the first population. If no checkpoint file is available, please set to an empty path.
CH4_sensitivities.csv # Sensitivities file name: if this path is empty, sensitivities for reactions will be computed and saved to this path. Otherwise, the program will load and use sensitivity values in the file for preprocessing directly.
CH4_Reduction_optimized_final.csv # Output checkpoint file name
0.0001 # Delta: accuracy tolerance during preprocessing with sensitivity
CH2CH2OH, CH3CHO, CH3CHOH, CH3CH2O, CH2CHOO, CH2CH2O2H, O2C2H4OH, nC3H7, iC3H7, C3H8, C4H4, C4, C4H, C4H2, H2CCCCH, i-C4H5, C4H6, nC4H7, C4H8, iC4H8 # Non-important-species: please manually choose more than 3 non-important-species that will be used for initialization
```

2. **Cantera Mechanism File**: A file describing the detailed mechanism to reduce. This file can be created through CHEMKIN files using the Cantera command `ck2yaml`.

Example command:

ck2yaml --input=MECH_FILE_NAME.inp --permissive --thermo=THERMO_FILE_NAME.dat


3. **Experimental IDT Data CSV File**

4. **Configuration File for Reading Experimental IDT Data**

Example file for reading `source_files\CH4_experimental_IDT.csv`:

```
54 # Number of RRCs to optimize: please do not modify this if you only do reduction
4 # Number of initial species: this number must match the experimental IDT data CSV file. Here, 4 refers to N2, Ar, O2, and CH4
13 # Number of groups in the experimental IDT data CSV file
3 # Maximum number of points in a group in the experimental IDT data CSV file
TUM_syngas_0.2.cti # Mech Name: a placeholder, please do not modify
OH # IDT peak species: please do not modify
```

Other input files listed in the general configuration file are not user-provided and will be created by the program.

## Getting Started

1. Ensure that all required input files are placed in the `\source_files\` directory.
2. Make sure that Python is installed properly on your device.
3. If you are starting the program for the first time, run `install.bat` to install all dependencies.
4. Start the reduction process by running `launch_reduction.bat`.

That's it! The program will now begin reducing the number of species in the combustion mechanism using a genetic algorithm.

## Code Architecture

The code is organized into five `.py` files:

1. `Data.gas.py`
2. `Data.Labels.py`
3. `Data.learnable_parameters.py`
4. `Tools.tools.py`
5. `GAPar.py`

`GAPar.py` defines the kernel control module of the program, the `BaseGA` class, and two subclasses.

The `GeneticAlgorithmForOptimization` subclass is for RRC optimization (inherited from a previous version). You do not need to read this part, as it is not relevant to the mechanism reduction process. Instead, focus on the `GeneticAlgorithmForReduction` subclass.

To perform the genetic algorithm on mechanisms, the following components must be defined:

- **Cantera object for mechanisms**: Implemented by the `Gas` class in `Data.gas.py`. This implementation style for the interface between Python and Cantera is due to the inability to transport Cantera objects between different processes.
- **Optimization targets**: Implemented by the `LearnableParameter` class in `Data.learnable_parameters.py`. Genetic algorithm operators are also implemented here.
- **Experimental data**: Implemented by the `Label` class in `Data.Labels.py`.

`Tools.tools.py` contains kernel methods that compute fitness values for each chromosome and other helper functions.

### Input Files for Mechanism Reduction

Each class mentioned above (except for `Tools.tools.py`) reads at least one input file. It's recommended to check the in-text comments for more information on the specific files.
