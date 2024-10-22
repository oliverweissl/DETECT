from __future__ import annotations
from dataclasses import dataclass, field
from collections import UserList

@dataclass
class MixCandidate:
    """A simple container for candidate elements used in style mixing."""

    label: int  # The class label of the candidate.
    is_w0: bool = False # Whether candidate is used for w0 calculation.
    weight: float = 1.  # The weight of the candidate for w0 calculation.
    w_index: int | None = None # Index in the w calculation.


class CandidateList(UserList):
    """
    A custom list like object to handle MixCandidates easily.

    Note this list object is immutable and caches getters.
    """
    _weights: list[float] | None
    _labels: list[int] | None
    _w_indices: list[int] | None
    _w0_candidates: CandidateList
    _wn_candidates: CandidateList

    def __init__(self, *initial_candidates: MixCandidate):
        super().__init__(initial_candidates)
        """If there are elements that have no index in the original collection we assign them to ensure persistent order."""
        max_i = -1
        for i, candidate in enumerate(self.data):
            if not candidate.w_index:
                candidate.w_index = max(i, max_i+1)
            elif candidate.w_index <= max_i:
                raise KeyError(f"Something corrupted the order of this Candidate List: {self._w_indices}")
            max_i = candidate.w_index

        if not any((elem.is_w0 for elem in self.data)):  # If none of candidates are w0 we take first candidate as w0.
            self.data[0].is_w0 = True

        self._weights = [elem.weight for elem in self.data]
        self._labels = [elem.label for elem in self.data]
        self._w_indices = [elem.w_index for elem in self.data]

    @property
    def weights(self) -> list[float]:
        return self._weights

    @property
    def labels(self) -> list[int]:
        return self._labels

    @property
    def w_indices(self) -> list[int]:
        return self._w_indices

    @property
    def w0_candidates(self) -> CandidateList:
        if not self._w0_candidates:
            self._w0_candidates = CandidateList(*[elem for elem in self.data if elem.is_w0])
        return self._w0_candidates

    @property
    def wn_candidates(self) -> CandidateList:
        if not self._wn_candidates:
            self._wn_candidates = CandidateList(*[elem for elem in self.data if not elem.is_w0])
        return self._wn_candidates

    """Make the list immutable."""
    def insert(self, index=None, value=None):
        raise TypeError()

    __setitem__ = insert
    __delitem__ = insert
    append = insert
    extend = insert
    pop = insert
    reverse = insert
    sort = insert
