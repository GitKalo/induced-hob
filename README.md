# Higher-order behaviors example

Supporting code for the paper _Kaloyan Danovski, Sandro Meloni, Michele Starnini, "Cross-order induced behaviors in contagion dynamics on higher-order networks"_.

This repository contains:
- The main `hob_example.ipynb` notebook, showing a reduced version of the dynamics simulation, measure computation, and data analysis.
- An auxiliary script of function definitions `utils.py`.
- An `environment.txt` file that can be used to set up your python environment using `pip` (mainly you need the [`hypergraphx`](https://github.com/HGX-Team/hypergraphx) package).

For the experiments in the paper, we generate the data with the same method shown in the example, but apply a scirpt for distributed compitation on a SLURM-enabled HPC cluster at the Pompeu Fabra University. These are also available upon request.