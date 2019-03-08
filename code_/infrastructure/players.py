import logging
import os
import pandas as pd

from lxml import etree
from os.path import join

from code_.infrastructure.game import Game
import settings as stg


class Players():
    """Soccer players.

    Attributes
    ----------
    all_players: pandas.DataFrame
    major_players: pandas.DataFrame
        Players having played more than 800 minutes in first half of season

    """

    def __init__(self,
                 filename='Noms des joueurs et IDs - F40 - L1 20162017.xml'):
        """Init class.

        Parameters
        ----------
        filename: string
            Filename containing all teams information and players

        """
        self.all_players = self._get_all_players_from_file(filename=filename)
        self.major_players = self._get_major_players()

    def _get_all_players_from_file(self, filename):
        tree = etree.parse(join(stg.DATA_DIR, filename))
        all_players = pd.DataFrame()

        for team in tree.xpath(stg.XML_PATH_TO_TEAMS):
            df_team = self._get_players_from_team(team_xml_element=team)\
                          .assign(**{stg.TEAM_NAME_COL: team.get(stg.XML_CLUB_NAME_ATTRIBUTE),
                                     stg.PLAYER_COL: lambda x: x[stg.PLAYER_COL].apply(lambda value: value.replace('p', ''))})

            all_players = pd.concat([all_players, df_team],
                                    axis=0, ignore_index=True)

        return all_players

    def _get_players_from_team(self, team_xml_element):
        players = team_xml_element.findall(stg.XML_PLAYER_TAG)

        df_team = pd.DataFrame({
            stg.PLAYER_COL: list(map(lambda x: x.get(stg.XML_PLAYER_ID_ATTRIBUTE), players)),
            stg.NAME_COL: list(map(lambda x: x.find(stg.XML_PLAYER_NAME_TAG).text, players)),
            stg.POSITION_COL: list(map(lambda x: x.find(stg.XML_PLAYER_POSITION_TAG).text, players))
        })
        return df_team

    def _get_major_players(self):
        logging.debug('Start computing players time spent on pitch..')
        number_of_processed_files = 0

        df_players = pd.DataFrame()

        for file_ in os.listdir(stg.GAMES_DIR):
            play_time = Game(filename=file_).get_play_time_by_player()
            df_players = pd.concat([df_players, play_time],
                                   axis=0, ignore_index=True)
            number_of_processed_files += 1
            logging.debug('.. {}/{} - Sucessfully loaded file {}'
                          .format(number_of_processed_files, len(os.listdir(stg.GAMES_DIR)), file_))

        major_players = df_players.groupby(stg.PLAYER_COL, as_index=False)\
                                  .agg({stg.GAME_TIME_COL: 'sum'})\
                                  .query('{} >= 60*800'.format(stg.GAME_TIME_COL))

        return major_players


if __name__ == '__main__':
    players = Players()
