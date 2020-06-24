import logging
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd

from scipy.stats import entropy

from ..utils.utils import Utils
from .measurement import Measurement


def get_entropy_func(opt):
    if opt.lower() == 'shannon':
        return Entropy.__shannon_entropy__
    elif opt.lower() == 'random':
        return Entropy.__random_entropy__
    else:
        raise ValueError('Undefined entropy function.')

def compute(df, features, options):
    for feat in features:
        df[f"{feat}_unique"] = df[feat].apply(set).apply(len)
        df[f"{feat}_count"]  = df[feat].apply(len)

    for feat in features:
        for opt in options:
            df[f"{feat}_entropy_{opt}"] = df[feat].apply(get_entropy_func(opt))

    return df


class Entropy(Measurement):
    """Computes the various Entropy measures."""

    ENTROPY_OPTIONS = ('shannon', 'random')


    def calculate(self, df_input, **kwargs):
        """
        Computes the Random or Shannon Entropy on selected features.

        Parameters
        ----------
        df_input:
            An input dataframe.
        options: list
            A list of entropy types, the random and shannon is selectable, and it takes both of them as default.
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
            options  = kwargs.get('options', Entropy.ENTROPY_OPTIONS)
            group    = kwargs.get('groupby', 'user')
            features = kwargs.get('features', ['domain'])
            workers  = kwargs.get('workers', 1)

            self.df = df_input \
                        .groupby(group, as_index=False) \
                        .agg({ f: list for f in features })

            if workers < 2:

                self.df = compute(self.df, features, options)

            else:

                df_split = np.array_split(self.df, workers)
                with mp.Pool(workers) as pool:
                    self.df = pd.concat(pool.map(partial(compute, features=features, options=options), df_split))


            self.df = self.df.drop(columns=[f for f in features])

            return self.df

        except Exception as e:
            logging.error(e)


    def __get_entropy_function__(self, opt):
        if opt.lower() == 'shannon':
            return self.__shannon_entropy__
        elif opt.lower() == 'random':
            return self.__random_entropy__
        else:
            raise ValueError('Undefined entropy function.')


    @staticmethod
    def __random_entropy__(seq):
        try:
            return np.log2(pd.Series(seq).unique().size) if len(seq) > 1 else None
        except Exception as e:
            logging.error('__random_entropy__',e)


    @staticmethod
    def __shannon_entropy__(seq):
        try:
            return entropy(pd.Series(seq).value_counts(), base=2) if len(seq) > 1 else None
        except Exception as e:
            logging.error('__shannon_entropy__', e)
