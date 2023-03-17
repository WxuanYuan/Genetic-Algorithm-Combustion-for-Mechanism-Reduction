This is a short documentation to help Jaro read the code a little.
It could be helpful to read the code following the logic below. :)

PS. The current code has not been cleared, please ignore the redundant variable and the chinese in-text comments.

Code Architecture
-----------
Generally, the code distributed in five [.py] files.

a. Data.gas.py

b. Data.Labels.py

c. Data.learnable_parameters.py

d. Tools.tools.py

e. GAPar.py

GAPar.py defines the kernel control module of the program, the BaseGA class and two subclasses.

Subclass GeneticAlgorithmForOptimization is for RRC optimization (that is inherited from history version), 
you do not have to read it. You only need to read another subclass GeneticAlgorithmForReduction.

To perform genetic algorithm on mechanisms, one has to define the following components:

(a) The Cantera object for mechanisms, this is done by class Gas defined in Data.gas.py. This wired way of implementing
the interface between Python and Cantera is because that one cannot transport Cantera objects through different processes.

(b) The optimization targets (e.g. the reduction code), this is done by class LearnableParameter defined in Data.learnable_parameters.py.
GA operators are also implemented here.

(c) The experimental data, this is done by class Label defined in Data.Labels.py

Tools.tools.py stores the kernel methods that computes fitness value for each chromosome and some other helping functions.


Input files for mechanism reduction
-----------
Each class mentioned above (except for Tools.tools.py) reads at least one input file, it's better to see the in-text comment.


Testing
-----------
Either UI or command line has not been implemented yet.
The only way to test the code is through mytest.py.

