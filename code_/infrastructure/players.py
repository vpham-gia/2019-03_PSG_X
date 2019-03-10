import logging
import numpy as np
import os
import pandas as pd

from lxml import etree
from os.path import join

from code_.infrastructure.game import Game
import settings as stg
stg.enable_logging(log_filename='players.log')


class Players():
    """Soccer players.

    Attributes
    ----------
    all_players: pandas.DataFrame
    transfered_players: pandas.DataFrame
    players_more_800_min: pandas.DataFrame
        Players having played more than 800 minutes in first half of season

    """

    def __init__(self, xml_filename=stg.FILENAME_ALL_PLAYERS,
                 saved_csv_filename=stg.FILENAME_PLAYERS_MORE_800):
        """Init class.

        Parameters
        ----------
        xml_filename: string
            Filename containing raw information all teams information and players
        saved_csv_filename: string
            Filename to filter players having played more than 800 minutes

        """
        self.all_players = self._get_all_players_from_file(xml_filename=xml_filename)
        self.transfered_players = self._get_transfered_players(xml_filename=xml_filename)
        self.players_more_800_min = self._get_players_more_800_min(read_from=saved_csv_filename)

    def get_eligible_players_list(self):
        """Filter players with 800+ min, not transfered neither loaned in January.

        Returns
        -------
        eligible_players: list

        """
        ids_800_min = self.players_more_800_min[stg.PLAYER_COL].apply(str).tolist()
        ids_transfered = self.transfered_players[stg.PLAYER_COL].apply(str).tolist()
        ids_loaned_to_exclude = self.all_players\
                                    .query('{} == "1"'.format(stg.LOAN_COL))\
                                    .query('{} >= "2017-01-01"'.format(stg.ARRIVAL_DATE))[stg.PLAYER_COL].apply(str).tolist()

        eligible_players = list(set(ids_800_min) - set(ids_transfered) - set(ids_loaned_to_exclude))
        return eligible_players

    def _get_all_players_from_file(self, xml_filename):
        tree = etree.parse(join(stg.DATA_DIR, xml_filename))
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
            stg.LOAN_COL: list(map(lambda x: x.get(stg.XML_LOAN_ATTRIBUTE), players)),
            stg.NAME_COL: list(map(lambda x: x.find(stg.XML_PLAYER_NAME_TAG).text, players)),
            stg.POSITION_COL: list(map(lambda x: x.find(stg.XML_PLAYER_POSITION_TAG).text, players)),
            stg.ARRIVAL_DATE: list(map(lambda x: self._get_detailed_info_for_player(x)[stg.XML_PLAYER_JOIN_DATE], players)),
            stg.LEAVE_DATE: list(map(lambda x: self._get_detailed_info_for_player(x)[stg.XML_PLAYER_LEAVE_DATE], players))
        })
        return df_team

    def _get_detailed_info_for_player(self, player_xml_element):
        dict_stat_text = {
            el.get(stg.XML_PLAYER_STAT_TYPE_ATTRIBUTE): el.text
            for el in player_xml_element.findall(stg.XML_PLAYER_STAT_TAG)
        }
        dict_stat_text.setdefault(stg.XML_PLAYER_JOIN_DATE, np.nan)
        dict_stat_text.setdefault(stg.XML_PLAYER_LEAVE_DATE, np.nan)
        return dict_stat_text

    def _get_transfered_players(self, xml_filename):
        tree = etree.parse(join(stg.DATA_DIR, xml_filename))
        transfered_players = pd.DataFrame()

        for team in tree.xpath(stg.XML_PATH_TO_TRANSFERS):
            df_team = self._get_players_from_team(team_xml_element=team)\
                          .assign(**{stg.TEAM_NAME_COL: team.get(stg.XML_CLUB_NAME_ATTRIBUTE),
                                     stg.PLAYER_COL: lambda x: x[stg.PLAYER_COL].apply(lambda value: value.replace('p', ''))})

            transfered_players = pd.concat([transfered_players, df_team],
                                           axis=0, ignore_index=True)

        return transfered_players

    def _get_players_more_800_min(self, read_from):
        try:
            return pd.read_csv(join(stg.OUTPUTS_DIR, read_from))
        except FileNotFoundError:
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

            players_more_800_min = df_players.groupby(stg.PLAYER_COL, as_index=False)\
                                             .agg({stg.GAME_TIME_COL: 'sum'})\
                                             .query('{} >= 60*800'.format(stg.GAME_TIME_COL))

            return players_more_800_min


if __name__ == '__main__':
    players = Players()
    toto = players.get_eligible_players_list()
