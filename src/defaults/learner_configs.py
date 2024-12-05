"""A collection of default parameters for the learner classes."""

from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS

REV_DE_DEFAULT_PARAMS = {
    "f": 0.9,
    "cr": 0.5,
    "bounds": (0, 1),
    "continuous": True,
    "population_size": 30,
}

GENETIC_DEFAULT_PARAMS = {
    "population_size": 30,
    "mutation_rate": 0.05,
}

PYMOO_DE_DEFAULT_PARAMS = {
    "bounds": (-100, 100),
    "algorithm": DE,
    "algo_params": {
        "pop_size": 100,
        "sampling": LHS(),
        "CR": 0.9,
        "F": 0.8,
    },
}

PYMOO_NSGA2_DEFAULT_PARAMS = {
    "bounds": (0, 1),
    "algorithm": NSGA2,
    "algo_params": {
        "pop_size": 100,
        "sampling": LHS(),
    },
}

PYMOO_AGE_MOEA_DEFAULT_PARAMS = {
    "bounds": (0, 1),
    "algorithm": AGEMOEA2,
    "algo_params": {
        "sampling": LHS(),
    },
}
