from typing import Any, Type, Union

import numpy as np
from numpy.typing import NDArray
from pymoo.core.algorithm import Algorithm
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.termination import NoTermination
from pymoo.problems.static import StaticProblem

from ._learner import Learner
from .auxiliary_components import OptimizerCandidate


class PymooLearner(Learner):
    """A Learner class for easy Pymoo integration"""

    _pymoo_algo: Algorithm
    _problem: Problem
    _pop_current: Population
    _bounds: tuple[int, int]

    def __init__(
        self,
        n_var: int,
        bounds: tuple[int, int],
        algorithm: Type[Algorithm],
        algo_params: dict[str, Any],
        num_objectives: int,
    ) -> None:
        """
        Initialize the genetic learner.

        :param n_var: The number of variables in the problem (mixing dimensions targeted).
        :param bounds: Bounds for the optimizer.
        :param algorithm: The pymoo Algorithm.
        :param algo_params: Parameters for the pymoo Algorithm.
        :param num_objectives: The number of objectives the learner can handle.
        """
        self._pymoo_algo = algorithm(**algo_params, save_history=True)
        self._n_var = n_var

        self._bounds = lb, ub = bounds
        self._problem = Problem(n_var=n_var, n_obj=num_objectives, xl=lb, xu=ub, vtype=float)
        self._pymoo_algo.setup(self._problem, termination=NoTermination())

        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._normalize_to_bounds(self._pop_current.get("X"))

        self._best_candidates = [
            OptimizerCandidate(
                solution=np.random.uniform(high=ub, low=lb, size=n_var), fitness=[np.inf] * num_objectives
            )
        ]
        self._previous_best = self._best_candidates.copy()

        self._learner_type = type(self._pymoo_algo)
        self._num_objectives = num_objectives

    def new_population(self) -> None:
        """
        Generate a new population.
        """
        static = StaticProblem(self._problem, F=np.column_stack(self._fitness))
        Evaluator().eval(static, self._pop_current)
        self._pymoo_algo.tell(self._pop_current)

        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._normalize_to_bounds(self._pop_current.get("X"))

    def get_x_current(self) -> tuple[Union[NDArray, None], NDArray]:
        """
        Return the current population in specific format.

        :return: The population as array of smx indices and smx weights.
        """
        # TODO: for now only one element can be used to mix styles -> should be n elements.
        smx_cond = np.zeros_like(self._x_current)
        smx_weights = self._x_current
        return smx_cond, smx_weights
