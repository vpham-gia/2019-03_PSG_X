import logging
import pandas as pd
from os.path import join

from code_.infrastructure.game import Game
from code_.infrastructure.players import Players
import settings as stg


class GameAnalyzer():
    """Class to analyze a game and build datasets for ML.

    Attributes
    ----------
    game: pandas.DataFrame
        Output of code_.infrastructure.game.Game.clean_game_data()
    major_players: list
        List of players having played at least 800 minutes

    """

    def __init__(self, **kwargs):
        """Initialize class."""
        try:
            self.game = self._fillna_game_data(**kwargs)
            logging.debug('Sucessfully initialized GameAnalyzer.')
        except TypeError as error:
            raise NameError('Error in GameAnalyzer initialization - {}'.format(error))

        self.major_players = self._get_list_major_players()

    def _fillna_game_data(self, **kwargs):
        game_data = Game(**kwargs).clean_game_data()
        game_data.fillna(value={stg.KEYPASS_COL: '0', stg.ASSIST_COL: '0'},
                         inplace=True)
        return game_data.dropna().reset_index(drop=True)

    def _get_list_major_players(self, filename=stg.FILENAME_PLAYERS_MORE_800):
        try:
            df_players = pd.read_csv(filepath_or_buffer=join(stg.OUTPUTS_DIR, filename))
            return df_players[stg.PLAYER_COL].values
        except FileNotFoundError:
            logging.info('CSV with major players not found - Computing Players method')
            return Players().major_players[stg.PLAYER_COL].values


if __name__ == '__main__':
    ga = GameAnalyzer(filename='f24-24-2016-853139-eventdetails.xml')
