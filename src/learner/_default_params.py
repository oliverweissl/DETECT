"""A collection of default parameters for the learner classes."""

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
    "bounds": (0, 1),
    "algorithm": DE,
    "algo_params": {
        "pop_size": 100,
        "sampling": LHS(),
        "cr": 0.3,
    },
}
