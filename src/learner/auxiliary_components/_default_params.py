"""A collection of default parameters for the learner classes."""

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS

REV_DE_DEFAULT_PARAMS = {
    "f": 0.9,
    "cr": 0.5,
    "bounds": (0, 1),
    "continuous": True,
    "population_size": 30,
    "num_objectives": 1,
}

GENETIC_DEFAULT_PARAMS = {
    "population_size": 30,
    "mutation_rate": 0.05,
    "num_objectives": 1,
}

PYMOO_DE_DEFAULT_PARAMS = {
    "bounds": (-100, 100),
    "algorithm": DE,
    "num_objectives": 1,
    "algo_params": {
        "pop_size": 100,
        "sampling": LHS(),
        "CR": 0.9,
        "F": 0.8,
    },
}
