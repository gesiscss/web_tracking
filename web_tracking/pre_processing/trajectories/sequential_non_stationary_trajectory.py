import logging

from .trajectory import Trajectory


class SequentialNonStationaryTrajectory(Trajectory):
    """Creates a trajectory over consecutively grouped domains."""

    COLUMNS = ['uid', 'user', 'domain', 'category', 'starts', 'ends', 'active_seconds', 'gap_seconds']

    def __init__(self, raw, validation_enabled=False, **kwargs):
        """Inherits the parent class constructor, and applies a validation.

        Parameters
        ----------
        raw:
            An input dataframe.
        validation_enabled: boolean,
            If true, checks if input dataframe is correctly provided.

        Returns
        -------
        void
        """
        super().__init__(raw)
        if validation_enabled:
            self.validate_input()


    def create(self, **kwargs):
        """Creates a domain-aggregated trajectory per user in tabular form.

        Parameters
        ----------
        threshold: int
            The seconds input which will be used for aggregation. To aggregate, gap seconds should be under threshold.
        features: [string], default is ['domain']
            Array of features that we create trajectories from.

        Returns
        -------
        pandas.DataFrame
            A trajectory
        """
        try:
            self.dfs = {}
            # check arguments
            self.threshold = kwargs.get('threshold', None)
            self.features = kwargs.get('features', ['domain'])

            if self.threshold is None:
                raise ValueError('Please give a threshold value.')
            # processing
            for feat in self.features:
                temp = self.raw.copy() \
                            .assign(
                                is_user_changed     = lambda df: df['user'] != df['user'].shift(),
                                is_feat_changed     = lambda df: df[feat] != df[feat].shift(),
                                is_threshold_passed = lambda df: df['gap_seconds'] > self.threshold,
                                can_be_aggregated   = lambda df: df['is_user_changed'] | df['is_feat_changed'] | df['is_threshold_passed'],
                                feat_group          = lambda df: df['can_be_aggregated'].cumsum()
                            )
                # grouping
                temp = temp \
                            .groupby('feat_group', as_index=False) \
                            .agg({
                                'uid'     : list,
                                'user'    : 'first',
                                feat      : 'first',
                                'starts'  : 'first',
                                'ends'    : 'last',
                            }) \
                            .rename(columns={'uid': 'uids'})
                # re-calculate values
                temp = temp \
                            .assign(
                                cumulative_active_seconds = lambda df: (df['ends'] - df['starts']).dt.total_seconds().apply(int),
                                gap_seconds               = lambda df: (df['starts'] - df['ends'].shift()).dt.total_seconds().fillna(0).apply(int) * (df['user'] != df['user'].shift())
                            )
                self.dfs[feat] = temp

            return self.dfs

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
        self.check_columns(DomainAggregatedTrajectory.COLUMNS, self.raw.columns)
        self.check_dtypes(self.raw, [('uid', 'any'),
                                     ('user', 'object'),
                                     ('starts', 'datetime'),
                                     ('ends', 'datetime'),
                                     ('active_seconds', 'numeric'),
                                     ('gap_seconds', 'numeric')])
