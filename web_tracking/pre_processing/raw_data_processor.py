import logging

import pandas as pd

from ..utils.validator import Validator


class RawDataProcessor(Validator):
    """A class for making pre-processing actions on the raw data."""

    def __init__(self, raw, *args):
        """A constructor of a class that requires a dataframe.

        Parameters
        ----------
        raw:
            An input dataframe which will be used in the further processes.

        Returns
        -------
        void
        """
        self.raw = raw


    def create_time_series(self, uid, user, domain, starts, url=None, category=None, ends=None, active_seconds=None):
        """It maps the raw data columns to the model dataframe and creates a ordered dataframe based on a time series, also it adds new columns.

        Parameters
        ----------
        raw : pandas.DataFrame
            A dataframe of the browsing records.
        uid : str
            A unique column identifier for the visits.
        user: str
            A unique identifier of a user.
        starts: str
            The starting timestamp identifier of a visit.
        ends: str, optional
            The ending timestamp identifier of a visit. Either this column or the active_seconds column should be provided.
        active_seconds: str, optional
            An identifer of a column about visit durations. Either this column or the ends column should be provided.

        Returns
        -------
        pandas.DataFrame
            A reorganized, sorted dataframe.
        """
        try:
            # check data
            # self.check_dtypes(self.raw, [(uid, 'any'), (user, 'any'), (url, 'object'), (domain, 'object'), (starts, 'datetime'), (category, 'object')])
            # map data
            self.df = self.raw \
                          .rename(columns={
                              uid     : 'uid',
                              user    : 'user',
                              domain  : 'domain',
                              starts  : 'starts'
                          }) \
                          .sort_values(['user', 'starts'])

            if category is not None:
                self.df = self.df.rename(columns={category: 'category'})

            if url is not None:
                self.df = self.df.rename(columns={url: 'url'})

            if ends is not None:
                # check data
                # self.check_dtypes(self.df, [(ends, 'datetime')])
                # map data
                self.df = self.df \
                              .rename(columns={
                                  ends: 'ends'
                              }) \
                              .assign(
                                  active_seconds = lambda df: (df['ends'] - df['starts']).dt.total_seconds()
                              )

            elif active_seconds is not None:
                # check data
                # self.check_dtypes(self.df, [(active_seconds, 'numeric')])
                # map data
                self.df = self.df \
                              .rename(columns={
                                  active_seconds: 'active_seconds'
                              }) \
                              .assign(
                                  ends = lambda df: df['starts'] + df['active_seconds'].apply(lambda x: pd.Timedelta(seconds=x)),
                              )

            else:
                raise ValueError("The attribute 'ends' or 'active_seconds' value must be given.")

            self.df = self.df \
                           .assign(
                               gap_seconds = lambda df: (df['starts'] - df['ends'].shift()).dt.total_seconds().fillna(0).apply(int) * (df['user'] == df['user'].shift())
                           )
        except Exception as e:
            logging.error(e)


    def normalize(self, data, numerical_cols=None, categorical_cols=None, **kwargs):

        if numerical_cols:
            for col in numerical_cols:
                if kwargs.get(col, None):
                    MIN, MAX = kwargs.get(col)[0], kwargs.get(col)[1]
                else:
                    MIN, MAX = data[col].min(), data[col].max()
                data.loc[:, f"{col}_normalized"] = data[col].apply(RawDataProcessor.__normalize_numerical__, args=(MIN, MAX,))

        if categorical_cols:
            for col in categorical_cols:
                data.loc[:, f"{col}_normalized"] = data[col].apply(RawDataProcessor.__normalize_categorical__, args=(kwargs.get(col),))

        return data


    @staticmethod
    def __normalize_numerical__(current, lowest, highest):
        current = current if current < highest else highest
        return (current - lowest) / (highest - lowest)


    @staticmethod
    def __normalize_categorical__(current, ordered_list_asc):
        try:
            INDEX = ordered_list_asc.index(current)
            MAX   = len(ordered_list_asc) - 1
            return INDEX / MAX
        except ValueError as e:
            return None



    # TODO: subdomain extraction
    def extract_subdomains(self, df):
        """Extracts the sub- and top-level domains from URL

        Parameters
        ----------
        df : pandas.DataFrame
            DF

        Returns
        -------
        pandas.DataFrame
            A dataframe with sub- and top-level domain columns
        """
        try:
            raise NotImplementedError('Please be patient, every method will be implemented... :)')
        except Exception as e:
            logging.error(e)
