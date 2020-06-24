import logging
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from .measurement import Measurement


def compute(df, features, options):

    for feat in features:
        for opt in options:
            df[f"{feat}_{opt}"] = df.apply(lambda r: Predictability.__pi__(r[f"{feat}_unique"], r[f"{feat}_entropy_{Predictability.PE[opt]}"]), axis=1)

    return df

class Predictability(Measurement):
    """Computes the predictability."""

    ENTROPY_OPTIONS = ('shannon', 'random', 'lz')
    PREDICTABILITY_OPTIONS = ('pi_unc', 'pi_rand', 'pi_max')
    PE = dict(zip(PREDICTABILITY_OPTIONS, ENTROPY_OPTIONS))


    def calculate(self, df, **kwargs):
        """Computes the Predictability on selected features.

        Parameters
        ----------
        df:
            An input dataframe.
        options: list
            A list of predictability types, uncorrelated(pi_unc), random(pi_rand) and maximum(pi_max) predictability is selectable. All is enabled as default
        features: list
            A list of columns which the predictability is calculated with.

        Returns
        -------
        pandas.DataFrame
            A dataframe with required predictability results.
        """
        try:
            options  = kwargs.get('options', Predictability.PREDICTABILITY_OPTIONS)
            features = kwargs.get('features', ['domain'])
            workers  = kwargs.get('workers', 1)

            if workers < 2:

                self.df = compute(df, features, options)

            else:

                df_split = np.array_split(df, workers)
                with mp.Pool(workers) as pool:
                    self.df = pd.concat(pool.map(partial(compute, features=features, options=options), df_split))

            return self.df

        except Exception as e:
            logging.error(e)


    @staticmethod
    def __pi__(unique_places, entropy):
        return fsolve(Predictability.__h__, 0.9999, xtol=10.**-20, args=(unique_places, entropy))[0] if unique_places > 1 else None

    @staticmethod
    def __h__(x, *args):
        try:
            n, s = args
            return -x*np.log2(x) - (1 - x)*(np.log2(1-x)) + (1 - x)*(np.log2(n-1)) - s
        except Exception as e:
            logging.error(e)
            return None
