import logging
import multiprocessing as mp

import numpy as np
import pandas as pd


class Utils:
    """General purpose functions."""

    @staticmethod
    def flatten(seq):
        """Flattens the nested list, only works on lists with depth=1

        Parameters
        ----------
        seq: list
            List of lists

        Returns
        -------
        list
            Flattened list.
        """
        try:
            return [item for sublist in seq for item in sublist]
        except Exception as e:
            logging.error(e)

    @staticmethod
    def parallelize_dataframe(df, func, cpu_count=mp.cpu_count()):
        """Executes the function in the parallel computing fashion.

        Parameters
        ----------
        df: pandas.DataFrame
            A dataframe.
        func: function
            A function which is applied on the dataframe.
        cpu_count: integer
            A number CPU that runs the method.

        Returns
        -------
        pandas.DataFrame
            A dataframe with computed results.
        """
        try:
            df_split = np.array_split(df, cpu_count)
            with mp.Pool(cpu_count) as pool:
                df = pd.concat(pool.map(func, df_split))
            return df
        except Exception as e:
            logging.error(e)
            return None
