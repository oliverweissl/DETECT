from abc import ABC, abstractmethod
from typing import Any

from ._criteria_kwargs import criteria_kwargs


class Criterion(ABC):
    """A criterion, allowing to evaluate events."""

    _name: str
    _inverse: bool

    def __init__(self, inverse: bool = False) -> None:
        """
        Initialize the criterion.

        :param inverse: Whether the criterion should be inverted.
        """
        self._inverse = inverse
        self._name = self._name + "Inv" if inverse else self._name

    @abstractmethod
    def evaluate(self, **kwargs: Any) -> float:
        """
        Evaluate the criterion in question.

        :param kwargs: The KW-Args parsed.
        :returns: The value(s).
        """
        ...
        # TODO: maybe return tuples always

    @property
    def name(self) -> str:
        """
        Get the criterions name.

        :returns: The name.
        """
        return self._name

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Automatically apply wrapper if function gets implemented.

        :param kwargs: The KW-Args parsed.
        """
        super().__init_subclass__(**kwargs)
        if "evaluate" in cls.__dict__:
            cls.evaluate = criteria_kwargs(cls.evaluate)
