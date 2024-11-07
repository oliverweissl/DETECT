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


class PymooLearner(Learner):
    """A Learner class for easy Pymoo integration"""

    _pymoo_algo: Algorithm
    _problem: Problem
    _pop_current: Population

    def __init__(
        self,
        n_var: int,
        bounds: tuple[int, int],
        algorithm: Type[Algorithm],
        algo_params: dict[str, Any],
    ) -> None:
        """
        Initialize the genetic learner.

        :param n_var: The number of variables in the problem (mixing dimensions targeted).
        :param bounds: Bounds for the optimizer.
        :param algorithm: The pymoo Algorithm.
        :param algo_params: Parameters for the pymoo Algorithm.
        """
        self._pymoo_algo = algorithm(**algo_params)

        lb, ub = bounds
        self._problem = Problem(n_var=n_var, n_obj=1, xl=lb, xu=ub, vtype=float)
        self._pymoo_algo.setup(self._problem, termination=NoTermination())

        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._pop_current.get("X")

        self._best_candidate = (None, np.inf)
        self._learner_type = type(self._pymoo_algo)

    def new_population(self) -> None:
        """
        Generate a new population.
        """
        static = StaticProblem(self._problem, F=self._fitness)
        Evaluator().eval(static, self._pop_current)
        self._pymoo_algo.tell(self._pop_current)

        self._pop_current = self._pymoo_algo.ask()
        self._x_current = self._pop_current.get("X")

    def get_x_current(self) -> tuple[Union[NDArray, None], NDArray]:
        """
        Return the current population in specific format.

        :return: The population as array of smx indices and smx weights.
        """
        smx_cond = np.zeros_like(
            self._x_current
        )  # TODO: for now only one element can be used to mix styles -> should be n elements.
        smx_weights = self._x_current
        return smx_cond, smx_weights
