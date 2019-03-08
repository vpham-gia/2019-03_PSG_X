import logging
# import pandas as pd
# from os.path import join

from code_.infrastructure.game import Game
# import settings as stg


class GameAnalyzer():
    """Class to analyze a game and build datasets for ML.

    Attributes
    ----------
    game: code_.infrastructure.game.Game object

    """

    def __init__(self, **kwargs):
        """Initialize class."""
        try:
            self.game = Game(**kwargs)
            logging.debug('Sucessfully initialized GameAnalyzer.')
        except TypeError as error:
            raise NameError('Error in GameAnalyzer initialization - {}'.format(error))


if __name__ == '__main__':
    ga = GameAnalyzer(filename='f24-24-2016-853139-eventdetails.xml')
