from typing import Any, Optional

import pandas as pd


class DefaultDF(pd.DataFrame):
    """Helper class to have uniform dataframes"""

    def __init__(self, pairs: bool = False, additional_fields: Optional[list[str]] = None) -> None:
        """
        Initialize the dataframe.

        :param pairs: Whether we store frontier pairs or not.
        :param additional_fields: Additional fields to add to the dataframe.
        """
        add = (
            ["X_prime_1", "y_prime_1", "X_prime_2", "y_prime_2"]
            if pairs
            else ["X_prime", "y_prime"]
        )
        af = additional_fields or []
        super().__init__(columns=["X", "y"] + add + ["runtime"] + af)

    def append_row(self, data: Any) -> None:
        """
        Append a single row at the end of the dataframe.

        :param data: The data to append.
        """
        self.loc[len(self)] = data
