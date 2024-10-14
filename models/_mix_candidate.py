from __future__ import annotations
from dataclasses import dataclass, field
from collections import UserList

@dataclass
class MixCandidate:
    """A simple container for candidate elements used in style mixing."""

    label: int  # The class label of the candidate.
    is_w0: bool  # Whether candidate is used for w0 calculation.
    weight: float = 1.  # The weight of the candidate for final w calculation.
    w_index: int | None = None # Index in the w calculation.


class CandidateList(UserList):
    """A custom list like object to handle MixCandidates easily."""
    def __init__(self, *initial_candidates: MixCandidate):
        super().__init__(initial_candidates)
        """If there are elements that have no index in the original collection we assign them to ensure persistent order."""
        max_i = -1
        for i, candidate in enumerate(self.data):
            if not candidate.w_index:
                candidate.w_index = max(i, max_i+1)
            elif candidate.w_index <= max_i:
                raise KeyError(f"Something corrupted the order of this Candidate List: {self.get_w_indices()}")
            max_i = candidate.w_index

    def get_weights(self) -> list[float]:
        return [elem.weight for elem in self.data]

    def get_labels(self) -> list[int]:
        return [elem.label for elem in self.data]

    def get_w_indices(self) -> list[int]:
        return [elem.w_index for elem in self.data]

    def get_w0_candidates(self) -> CandidateList:
        return CandidateList(*[elem for elem in self.data if elem.is_w0])
