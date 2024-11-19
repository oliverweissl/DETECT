from abc import ABC, abstractmethod
from typing import Any


class Criterion(ABC):
    """A criterion, allowing to evaluate events."""

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> float:
        """
        Evaluate the criterion in question.

        :param kwargs: The KW-Args parsed.
        :returns: The value(s).
        """
        ...
        # TODO: maybe return tuples always
