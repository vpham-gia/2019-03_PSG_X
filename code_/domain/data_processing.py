from os.path import join
import pandas as pd

import settings as stg


class DataQualityChecker():
    """Class to check data quality.

    Attributes
    ----------
    data: pandas.DataFrame

    """

    def __init__(self, filename):
        """Class init.

        Parameters
        ----------
        filename: string

        """
        self.data = pd.read_csv(join(stg.OUTPUTS_DIR, filename))

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
        print(df.head(10))


if __name__ == '__main__':
    dqc = DataQualityChecker(filename=stg.FILENAME_STATS_AGGREGATED)
    dqc.print_completeness()
    dqc.print_min_nb_observations_by_target(target=stg.PLAYER_COL)
