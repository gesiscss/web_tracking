import logging

from .trajectory import Trajectory
from .binned_stationary_trajectory import BinnedStationaryTrajectory

class BinnedNonStationaryTrajectory(Trajectory):
    """Creates a trajectory with consecutively non-repeating domains in a sequence."""

    COLUMNS = ['uid', 'user', 'domain', 'category']

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
        """Creates a non-stationary trajectory per user in tabular form. It does not have consecutive entries with same domain.

        Parameters
        ----------
        void

        Returns
        -------
        pandas.DataFrame
            A trajectory
        """
        try:
            self.dfs = {}
            # check arguments
            self.features = kwargs.get('features', ['domain'])
            self.bin_size = kwargs.get('bin_size', 60)

            BST = BinnedStationaryTrajectory(self.raw)
            BST.create(bin_size=self.bin_size)

            # check input dataframe
            # self.validate_input()
            # processing
            for feat in self.features:
                temp = BST.df.copy() \
                            .assign(
                                is_user_changed     = lambda df: df['user'] != df['user'].shift(),
                                is_feat_changed     = lambda df: df[feat] != df[feat].shift(),
                                can_be_aggregated   = lambda df: df['is_user_changed'] | df['is_feat_changed'],
                                feat_group          = lambda df: df['can_be_aggregated'].cumsum()
                            ) \
                            .groupby('feat_group', as_index=False) \
                            .agg({
                                'uid'     : list,
                                'user'    : 'first',
                                feat      : 'first',
                            }) \
                            .rename(columns={'uid': 'uids'})
                self.dfs[feat] = temp

            return self.dfs
            # check output dataframe
            # self.validate_output()

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
        self.check_columns(NonStationaryTrajectory.COLUMNS, self.raw.columns)
        self.check_dtypes(self.raw, [('uid', 'any'),
                                     ('user', 'object'),
                                     ('domain', 'object'),
                                     ('category', 'object')])


    def validate_output(self):
        """Executes the validation functions for the output trajectory.

        Parameters
        ----------
        void

        Returns
        -------
        void
        """
        self.check_dtypes(self.df, [('uids', 'list'),
                                    ('user', 'object'),
                                    ('domain', 'object'),
                                    ('category', 'object')])
