from typing import Any

import pandas as pd


class DefaultDF(pd.DataFrame):
    """Helper class to have uniform dataframes"""

    def __init__(self, pairs: bool = False) -> None:
        """
        Initialize the dataframe.

        :param pairs: Whether we store frontier pairs or not.
        """
        add = ["X_prime_1", "y_prime_1", "X_prime_2", "y_prime_2"] if pairs else ["X_prime", "y_prime"]
        super().__init__(columns=["X", "y"] + add + ["runtime"])

    def append_row(self, data: Any) -> None:
        """
        Append a single row at the end of the dataframe.

        :param data: The data to append.
        """
        self.loc[len(self)] = data
