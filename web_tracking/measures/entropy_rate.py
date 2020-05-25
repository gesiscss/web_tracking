import logging
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd

from .measurement import Measurement


def compute(df, features):
    for feat in features:
        df[f"{feat}_entropy_lz"] = df[feat].apply(EntropyRate.__lempel_ziv_estimate__)

    return df


class EntropyRate(Measurement):
    """A measurement class for Lempel-Ziv Entropy computation."""

    def __init__(self):
        self.COLUMNS = ['user', 'domain', 'category']

    def calculate(self, df_input, **kwargs):
        """Computes the Lempel-Ziv Entropy on selected features.

        Parameters
        ----------
        group: str
            A column for grouping, it is 'user' as default.
        features: list
            A list of columns which the entropy is calculated with.

        Returns
        -------
        pandas.DataFrame
            A dataframe with required entropy results.
        """
        try:
            group = kwargs.get('groupby', 'user')
            features = kwargs.get('features', ['domain'])
            workers  = kwargs.get('workers', 1)

            self.raw = df_input
            # self.validate_input()
            self.df = df_input \
                        .groupby(group, as_index=False) \
                        .agg({f: list for f in features})
            # self.check_dtypes(self.df, [(f"{f}", 'list') for f in features])

            if workers < 2:

                self.df = compute(self.df, features)

            else:

                df_split = np.array_split(self.df, workers)
                with mp.Pool(workers) as pool:
                    self.df = pd.concat(pool.map(partial(compute, features=features), df_split))

            # self.validate_output()
            # self.check_dtypes(self.df, [(f"{f}_entropy_lz", 'numeric') for f in features])

            self.df = self.df.drop(columns=[f for f in features])

            return self.df
        except Exception as e:
            logging.error(e)


    def validate_input(self):
        self.check_columns(self.COLUMNS, self.raw.columns)
        self.check_dtypes(self.raw, [('user', 'object'), ('domain', 'object'), ('category', 'object')])

    def validate_output(self):
        pass

    @staticmethod
    def __lempel_ziv_estimate__(seq):
        try:
            return EntropyRate.__lempel_ziv_complexity__(seq)/(len(seq) / np.log2(len(seq))) if len(seq) > 1 else None
        except IndexError as e:
            logging.error('__lempel_ziv_estimate__:', f'{e}: {str(seq)}')
            return None

    @staticmethod
    def __lempel_ziv_complexity__(seq):
        # this is basic http://sci-hub.tw/10.1103/PhysRevA.36.842
        c = 1
        l = 1
        i = 0
        k = 1
        k_max = 1
        stop = False
        while not stop:
            if seq[i + k - 1] != seq[l + k - 1]:
                if k > k_max:
                    k_max = k
                i += 1
                if i == l:
                    c += 1
                    l += k_max
                    if l + 1 > len(seq):
                        stop = True
                    else:
                        i = 0
                        k = 1
                        k_max = 1
                else:
                    k = 1
            else:
                k += 1
                if l + k > len(seq):
                    c += 1
                    stop = True
        return c
