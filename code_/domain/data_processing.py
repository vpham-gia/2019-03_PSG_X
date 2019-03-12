from os.path import join
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import settings as stg


class DataQualityChecker():
    """Class to check data quality.

    Attributes
    ----------
    data: pandas.DataFrame

    """

    def __init__(self, df):
        """Class init.

        Parameters
        ----------
        df: pandas.DataFrame

        """
        self.data = df

    def print_completeness(self):
        """Print information about data quality."""
        print('\n')
        print('Number of obserations: {}'.format(self.data.shape[0]))
        print('Number after dropna(): {}'.format(self.data.dropna().shape[0]))

    def print_min_nb_observations_by_target(self, target):
        """Print smallest nb of observations grouped by target."""
        df = self.data.dropna()\
                      .groupby(target)\
                      .agg({target: 'count'})\
                      .rename(columns={target: 'nb_observations'})\
                      .sort_values(by='nb_observations', ascending=True)
        print('\n')
        print(df.head(5))


class CategoricalProjector(BaseEstimator, TransformerMixin):
    """Project categorical variables using the target variable.

    Attributes
    ----------
    column_name: string
        Name of column to project using change column
    columns_to_build_change_var: list of length 2
        2 columns used to compute change column (boolean)
    projection_dict: dict
        Dictionnary to map values
    mean: float
        Mean value of train set to impute if new categories are found

    """

    def __init__(self, column_to_substitute, columns_to_build_change_var):
        """Init class."""
        self.column_name = column_to_substitute

        try:
            assert(len(columns_to_build_change_var) == 2)
            self.columns_to_build_change_var = columns_to_build_change_var
        except AssertionError:
            raise AttributeError('columns_to_build_change_var must have 2 elements')

        self.projection_dict = dict()
        self.mean = None

    def fit(self, X, y):
        """Fit."""
        CHANGE_COL = 'change'
        NB_EVENT_COL = 'nb_events'

        df = pd.concat([X, y], axis=1)
        df[CHANGE_COL] = (df[self.columns_to_build_change_var[0]] != df[self.columns_to_build_change_var[1]]).apply(str)

        self.projection_dict = df.groupby([self.column_name, CHANGE_COL])\
                                 .agg({CHANGE_COL: 'count'})\
                                 .rename(columns={CHANGE_COL: NB_EVENT_COL})\
                                 .reset_index()\
                                 .pivot(index=self.column_name, columns=CHANGE_COL, values=NB_EVENT_COL)\
                                 .fillna(value=0)\
                                 .assign(change_ratio=lambda x: x['True'] / (x['True'] + x['False']))\
                                 .drop(labels=['False', 'True'], axis=1)\
                                 .to_dict()['change_ratio']

        self.mean = X[self.column_name].map(self.projection_dict).mean()

        return self

    def transform(self, X):
        """Transform."""
        X[self.column_name] = X[self.column_name].map(self.projection_dict)\
                                                 .fillna(value=self.mean)
        return X


if __name__ == '__main__':
    dqc = DataQualityChecker(filename=stg.FILENAME_STATS_AGGREGATED)
    dqc.print_completeness()
    dqc.print_min_nb_observations_by_target(target=stg.PLAYER_COL)

    train = pd.DataFrame({'game': ['XX', 'YY', 'ZZ'],
                          'team_id': ['0', '0', '0'],
                          'period_id': ['1', '1', '1'],
                          'type_id_lag1': [1, 1, 4],
                          'team_id_lag1': ['1', '1', '0'],
                          'x_along_team1_axis_lag1': ['50', '50']})
    X_train, y_train = train[stg.NEXT_TEAM_FEATURES], train[stg.NEXT_TEAM_TARGET]

    df_ex = pd.DataFrame({'game': ['XX', 'YY'],
                          'team_id': ['0', '0'],
                          'period_id': ['1', '1'],
                          'type_id_lag1': [1, 100],
                          'team_id_lag1': ['0', '0'],
                          'x_along_team1_axis_lag1': ['50', '50']})
    toto = df_ex[stg.NEXT_TEAM_FEATURES]
    cat_proj = CategoricalProjector(column_to_substitute='{}_lag1'.format(stg.EVENT_TYPE_COL),
                                    columns_to_build_change_var=[stg.NEXT_TEAM_TARGET, '{}_lag1'.format(stg.NEXT_TEAM_TARGET)])
    cat_proj.fit_transform(X_train, y_train)
    cat_proj.transform(toto)
