import logging

import pandas.api.types as tp


class Validator:
    """A class for validating the dataframes."""

    def check_columns(self, required_columns, input_columns):
        """Checks if the required columns are in the input columns.

        Parameters
        ----------
        required_columns: list
            A list of column names that we want.
        input_columns: list
            A list of column names that we search in.

        Returns
        -------
        Boolean
            If required columns take place, True.
        """
        try:
            diff = set(required_columns).difference(set(input_columns))
            if len(diff) == 0:
                return True
            else:
                raise ValueError(f"Please give values of {','.join(diff)} for the mapping process.")
        except Exception as e:
            logging.error(e)


    def check_dtypes(self, df, rules):
        """Checks if the types of the dataframe columns.

        Parameters
        ----------
        df: pandas.DataFrame
            An input dataframe.
        rules: dict
            A dictionary of column name and type string pairs.

        Returns
        -------
        Boolean
            If columns have the required types, True.
        """
        try:
            if all(isinstance(item, tuple) for item in rules) and all(isinstance(item[0], str) and isinstance(item[1], str) for item in rules):
                for c, t in rules:
                    if not self.__get_validator_func__(t)(df[c]):
                        raise AssertionError(f"'{c}' column fails in the validation process, please check that column.")
                return True
            else:
                raise ValueError('Please give a list of tuples with string pair. The first value is for a column, and the second is for the type.')
        except Exception as e:
            logging.error(e)


    # TODO: consistency check: uniqueness, end - start > 0, active_seconds > 0, gap_seconds > 0, url regex, domain regex
    def check_consistency(self):
        pass


    def __get_validator_func__(self, t):
        """Returns a column type validator function.

        Parameters
        ----------
        t: string
            A type of column

        Returns
        -------
        function
            A function that validates the column.
        """
        if   t  == 'any'     : return lambda x: True
        elif t  == 'string'  : return tp.is_string_dtype
        elif t  == 'numeric' : return tp.is_numeric_dtype
        elif t  == 'datetime': return tp.is_datetime64_ns_dtype
        elif t  == 'object'  : return tp.is_object_dtype
        elif t  == 'list'    : return lambda s: all(s.apply(tp.is_list_like))
        elif t  == 'tuple'   : return lambda s: all(s.apply(tp.is_tuple_like))
        else                 : return lambda x: False
