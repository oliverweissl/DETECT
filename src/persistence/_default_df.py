from typing import Any

import pandas as pd


class DefaultDF(pd.DataFrame):
    """Helper class to have uniform dataframes"""

    def __init__(self, pairs: bool = False) -> None:
        """
        Initialize the dataframe.

        :param pairs: Whether we store frontier pairs or not.
        """
        super().__init__()
        add = (
            ["X_prime", "y_prime"]
            if not pairs
            else ["X_prime_1", "y_prime_1", "X_prime_2", "y_prime_2"]
        )
        self.columns = ["X", "y"] + add + ["runtime"]

    def append_row(self, data: Any) -> None:
        """
        Append a single row at the end of the dataframe.

        :param data: The data to append.
        """
        self.loc[len(self)] = data
