from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentConfig:
    """A simple Dataclass to store experiment configs."""

    samples_per_class: int  # How candidates should be searched for per class
    generations: int  # How many generations we search for candidates.

    classes: list[int]  # The classes in the experiment.
    save_to: Optional[str] = field(
        default=None
    )  # The name of the dataframe to save to, if None dont save.
