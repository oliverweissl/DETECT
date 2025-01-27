import pandas as pd
from typing import Any

class DefaultDF(pd.DataFrame):
    """Helper class to have uniform dataframes"""
    def __init__(self):
        super().__init__()
        self.columns = ["X", "y", "X_prime", "y_prime", "runtime"]

    def append_row(self, data: Any) -> None:
        """
        Append a single row at the end of the dataframe.

        :param data: The data to append.
        """
        self.loc[len(self)] = data
