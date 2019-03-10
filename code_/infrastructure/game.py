import pandas as pd

from lxml import etree
from os.path import join

import settings as stg


class Game():
    """Soccer game representation.

    Attributes
    ----------
    filename: string
    data: pandas.DataFrame
        All events recorded in file
    home_id: string
    away_id: string

    """

    def __init__(self, filename):
        """Initialize class.

        Parameters
        ----------
        filename: string

        """
        self.filename = filename
        self.data = self._gather_events()
        self.home_id, self.away_id = self._get_home_and_away_id()

    def get_play_time_by_player(self):
        """Build time played by every player.

        Returns
        -------
        df_play_time: pandas.DataFrame
            Dataframe with at least player_id and game_time_in_sec

        """
        end_first_half, end_second_half = self._get_end_of_periods()

        df_substitutions = self._get_substitutions()\
                               .fillna({stg.PLAYER_START_COL: 0,
                                        stg.PLAYER_END_COL: end_second_half})

        start_players = self._get_teams_start_list()
        full_game_players = set(start_players) - set(df_substitutions[stg.PLAYER_COL].values)
        df_full_game_players = pd.DataFrame({stg.PLAYER_COL: list(full_game_players)})\
                                 .assign(**{stg.PLAYER_START_COL: 0,
                                            stg.PLAYER_END_COL: end_second_half})

        df_play_time = pd.concat([df_full_game_players, df_substitutions],
                                 axis=0, ignore_index=True)
        df_play_time[stg.PLAYER_EXTRA_TIME_FIRST_HALF] = df_play_time.apply(lambda x: x[stg.PLAYER_START_COL] < 45 * 60 < x[stg.PLAYER_END_COL],
                                                                            axis=1)
        df_play_time[stg.GAME_TIME_COL] = \
            df_play_time[stg.PLAYER_END_COL] \
            - df_play_time[stg.PLAYER_START_COL] \
            + (end_first_half - 45 * 60) * df_play_time[stg.PLAYER_EXTRA_TIME_FIRST_HALF]
        return df_play_time

    def clean_game_data(self):
        """Perform data cleaning operations.

        Returns
        -------
        clean_data: pandas.DataFrame

        """
        game_data = self.data.copy()

        _first_second_half = self._keep_first_second_half(df=game_data)
        _time_in_sec = self._convert_time_to_seconds(df=_first_second_half)
        _filter_events = self._drop_events(df=_time_in_sec,
                                           events_id_list=[stg.EVENTS_MAP['DELETED_EVENT']])
        _add_game_teams = self._add_game_teams_index(df=_filter_events)
        _replace_teams_id = self._substitute_team_id(df=_add_game_teams)

        clean_data = self._keep_game_columns_if_exists(df=_replace_teams_id,
                                                       list_cols=stg.COLS_TO_KEEP)
        return clean_data

    def _get_teams_start_list(self):
        try:
            tree = etree.parse(join(stg.GAMES_DIR, self.filename))
            start_list_all = list()

            for event in tree.xpath(stg.XML_PATH_TO_EVENTS):
                if event.get(stg.EVENT_TYPE_COL) == stg.EVENTS_MAP['KICKOFF']:
                    # Could be refactored with .find method
                    for qualifier in event.getchildren():
                        if qualifier.get(stg.QUALIFIER_COL) == stg.QUALIFIER_MAP['ALL_PLAYERS']:
                            team = qualifier.get('value')
                            start_list = team.split(', ')[:11]
                            start_list_all.extend(start_list)

            assert(len(start_list_all) == 22)
            return start_list_all
        except AssertionError as error:
            raise Exception('Start list does not contain 22 players: {}'.format(error))

    def _get_substitutions(self):
        try:
            df_substitutions = self.data.query('{} in ["{}", "{}"]'.format(stg.EVENT_TYPE_COL,
                                                                           stg.EVENTS_MAP['PLAYER_OFF'],
                                                                           stg.EVENTS_MAP['PLAYER_ON']))
            df_time_sec = self._convert_time_to_seconds(df=df_substitutions)

            df_substitutions = df_time_sec.pivot(index=stg.PLAYER_COL,
                                                 columns=stg.EVENT_TYPE_COL,
                                                 values=stg.GAME_TIME_COL)\
                                          .rename(columns={stg.EVENTS_MAP['PLAYER_ON']: stg.PLAYER_START_COL,
                                                           stg.EVENTS_MAP['PLAYER_OFF']: stg.PLAYER_END_COL})\
                                          .reset_index()
            assert(df_substitutions.shape[0] <= 12)
            return df_substitutions
        except AssertionError as error:
            raise Exception('More substitution than allowed: {}'.format(error))

    def _get_end_of_periods(self):
        tree = etree.parse(join(stg.GAMES_DIR, self.filename))
        end_first_half, end_second_half = -1, -1

        for event in tree.xpath(stg.XML_PATH_TO_EVENTS):
            if event.get(stg.EVENT_TYPE_COL) == stg.EVENTS_MAP['END_PERIOD']:
                if event.get(stg.PERIOD_COL) == '1':
                    end_first_half = \
                        60 * int(event.get(stg.MINUTES_COL)) \
                        + int(event.get(stg.SECONDS_COL))
                elif event.get(stg.PERIOD_COL) == '2':
                    end_second_half = \
                        60 * int(event.get(stg.MINUTES_COL)) \
                        + int(event.get(stg.SECONDS_COL))

        return end_first_half, end_second_half

    def _keep_first_second_half(self, df):
        df_out = df.query('{} in ["1", "2"]'.format(stg.PERIOD_COL))
        return df_out

    def _convert_time_to_seconds(self, df):
        df_out = df.copy()
        df_out[stg.GAME_TIME_COL] = \
            60 * df_out[stg.MINUTES_COL].apply(int) + df_out[stg.SECONDS_COL].apply(int)
        return df_out

    def _drop_events(self, df, events_id_list):
        df_out = df.query('{} not in {}'.format(stg.EVENT_TYPE_COL,
                                                events_id_list))
        return df_out

    def _add_game_teams_index(self, df):
        df_out = df.copy()
        df_out[stg.GAME_ID_COL] = '{}-{}'.format(self.home_id, self.away_id)
        return df_out

    def _substitute_team_id(self, df):
        df_out = df.copy()
        df_out[stg.TEAM_COL].replace({self.home_id: '1', self.away_id: '0'},
                                     inplace=True)
        return df_out

    def _keep_game_columns_if_exists(self, df, list_cols):
        return df[[col for col in list_cols if col in df.columns]]

    def _gather_events(self):
        tree = etree.parse(join(stg.GAMES_DIR, self.filename))
        df_events = pd.DataFrame()

        for event in tree.xpath(stg.XML_PATH_TO_EVENTS):
            df_event = pd.DataFrame(dict(event.attrib), index=[0])
            df_events = pd.concat([df_events, df_event],
                                  axis=0, ignore_index=True, sort=False)

        return df_events

    def _get_home_and_away_id(self):
        tree = etree.parse(join(stg.GAMES_DIR, self.filename))
        game_info = tree.xpath(stg.XML_PATH_TO_GAME_INFO)[0]
        return game_info.get('home_team_id'), game_info.get('away_team_id')


if __name__ == '__main__':
    game = Game(filename='f24-24-2016-853139-eventdetails.xml')

    # df = game.clean_data()

    toto = game._get_teams_start_list()
    subs = game._get_substitutions()
    df = game.get_play_time_by_player()
    toto = game.clean_game_data()
