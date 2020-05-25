import logging

import pandas as pd

from ...utils.utils import Utils
from .trajectory import Trajectory


class BinnedStationaryTrajectory(Trajectory):
    """Creates a trajectory with equally sized bins."""

    COLUMNS = ['uid', 'user', 'domain', 'category', 'starts', 'ends', 'active_seconds']

    def __init__(self, raw, **kwargs):
        """Inherits the parent class constructor, and applies a validation.

        Parameters
        ----------
        raw:
            An input dataframe.

        Returns
        -------
        void
        """
        super().__init__(raw)
        # self.validate_input()


    def create(self, **kwargs):
        """Creates a binned trajectory.

        Parameters
        ----------
        bin_size: int
            The size of bin in seconds which is used for the binning.

        Returns
        -------
        pandas.DataFrame
            A trajectory
        """
        try:
            # check arguments
            self.bin_size = kwargs.get('bin_size', None)
            if self.bin_size is None:
                raise ValueError('Please give a bin size value.')
            # check input dataframe
            # self.validate_input()
            # processing
            starts_global = self.raw['starts'].min()
            temp = self.raw \
                        .assign(
                            bin_start_time = lambda df: (df['starts'] - starts_global).apply(lambda d: int(d.total_seconds())),
                            bin_end_time   = lambda df: (df['ends'] - starts_global).apply(lambda d: int(d.total_seconds())),
                            bin_start_num  = lambda df: (df['bin_start_time'] // self.bin_size) + 1,
                            bin_end_num    = lambda df: (df['bin_end_time'] // self.bin_size) + 1,
                        )
            # constructing
            self.df =  pd.DataFrame(
                            Utils.flatten(temp.apply(lambda r: self.__find_bins_in_url__(r, self.bin_size), axis=1)),
                            columns=['user', 'uid', 'bin', 'seconds', 'share']
                        ) \
                        .sort_values(['user', 'bin', 'seconds']) \
                        .drop_duplicates(['user', 'bin'], keep='last') \
                        .merge(temp[['user', 'uid', 'domain']], on=['user', 'uid'])

            return self.df
            # check output dataframe
            # self.validate_output()
        except Exception as e:
            logging.error(e)


    def __find_bins_in_url__(self, row, bin_size):
        """Splits up a visit entry into bins.

        Parameters
        ----------
        row: pandas.Series
            A visit entry
        bin_size: int
            The size of the bin.

        Returns
        -------
        bins: list
            A list of bin numbers
        """
        try:
            id = (row['user'], row['uid'])
            bins = list(range(row['bin_start_num'], row['bin_end_num']+1))
            if row['active_seconds'] < bin_size and len(bins) == 1:
                bins = list(map(lambda b: (*id, b, row['active_seconds'], row['active_seconds'] / bin_size), bins))
            else:
                f, *m, l = bins
                f_bin = f * bin_size - row['bin_start_time']
                l_bin = row['bin_end_time'] - (l-1) * bin_size
                bins = [
                    (*id, f, f_bin, f_bin / bin_size),
                    *list(map(lambda b: (*id, b, bin_size, bin_size / bin_size), m)),
                    (*id, l, l_bin, l_bin / bin_size)
                ]
            return bins
        except Exception as e:
            logging.error(e)


    def validate_input(self):
        """Executes the validation functions for the input trajectory.

        Parameters
        ----------
        void

        Returns
        -------
        void
        """
        self.check_columns(BinnedTrajectory.COLUMNS, self.raw.columns)
        self.check_dtypes(self.raw, [('uid', 'any'),
                                     ('user', 'object'),
                                     ('domain', 'object'),
                                     ('starts', 'datetime'),
                                     ('ends', 'datetime'),
                                     ('active_seconds', 'numeric')])


    def validate_output(self):
        """Executes the validation functions for the output trajectory.

        Parameters
        ----------
        void

        Returns
        -------
        void
        """
        self.check_dtypes(self.df, [('uid', 'any'),
                                    ('user', 'object'),
                                    ('domain', 'object'),
                                    ('category', 'object'),
                                    ('bin', 'numeric'),
                                    ('seconds', 'numeric'),
                                    ('share', 'numeric')])
