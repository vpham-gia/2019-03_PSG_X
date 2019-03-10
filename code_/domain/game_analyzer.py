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
    major_players: pandas.DataFrame
        Similar to Players().major_players

    """

    def __init__(self, **kwargs):
        """Initialize class."""
        try:
            self.game = self._fillna_game_data(**kwargs)
            logging.debug('Sucessfully initialized GameAnalyzer.')
        except TypeError as error:
            raise NameError('Error in GameAnalyzer initialization - {}'.format(error))

        self.major_players = self._get_df_major_players()

    def build_game_dataset_major_players(self, sliding_interval_min,
                                         list_events_number,
                                         list_events_with_sucess_rate):
        """Compute stats for every period with 15 min intervals for major players.

        Parameters
        ----------
        sliding_interval_min: integer
            Time interval to move sliding window
        list_events_number: list
            Event types ID for which number of events are computed
        list_events_with_sucess_rate: list
            Event types ID for which number of events and sucess rate are computed

        Returns
        -------
        df_game_stats: pandas.DataFrame
            Columns: game, player_id, binarized team_id, time_interval, stats

        """
        df_game_stats = pd.DataFrame()
        major_players_list = self.major_players[stg.PLAYER_COL].apply(str).values.tolist()

        for start in range(0, 36, sliding_interval_min):
            df = self._get_info_on_15_minutes(period='1', start_in_min=start,
                                              list_events_number=list_events_number,
                                              list_events_with_sucess_rate=list_events_with_sucess_rate)\
                     .query('{} in {}'.format(stg.PLAYER_COL, major_players_list))
            df_game_stats = pd.concat([df_game_stats, df], axis=0,
                                      ignore_index=True, sort=False)

        for start in range(45, 76, sliding_interval_min):
            df = self._get_info_on_15_minutes(period='2', start_in_min=start,
                                              list_events_number=list_events_number,
                                              list_events_with_sucess_rate=list_events_with_sucess_rate)\
                     .query('{} in {}'.format(stg.PLAYER_COL, major_players_list))
            df_game_stats = pd.concat([df_game_stats, df], axis=0,
                                      ignore_index=True, sort=False)

        return df_game_stats

    def _get_info_on_15_minutes(self, period, start_in_min, **kwargs):
        df_15_min = self.game.query('{start}*60 <= {col} <= ({start}+15)*60'.format(start=start_in_min,
                                                                                    col=stg.GAME_TIME_COL))\
                             .query('{} == "{}"'.format(stg.PERIOD_COL, period))

        try:
            df_players = df_15_min.filter(items=[stg.GAME_ID_COL, stg.PLAYER_COL, stg.TEAM_COL])\
                                  .drop_duplicates()\
                                  .assign(time_period='{}-{}'.format(start_in_min, start_in_min + 15))
            assert(df_players.shape[0] == len(df_players[stg.PLAYER_COL].unique()))
        except AssertionError:
            raise AssertionError('Some players belong to both teams.')

        df_player_stats = self._get_agg_stats(df_with_events=df_15_min, agg_column=stg.PLAYER_COL, **kwargs)
        df_player_stats.columns = ['p_{}'.format(col) if col != stg.PLAYER_COL else col
                                   for col in df_player_stats.columns]

        df_team_stats = self._get_agg_stats(df_with_events=df_15_min, agg_column=stg.TEAM_COL, **kwargs)
        df_team_stats.columns = ['t_{}'.format(col) if col != stg.TEAM_COL else col
                                 for col in df_team_stats.columns]

        df_all_stats = df_players.merge(right=df_player_stats, on=stg.PLAYER_COL, how='left')\
                                 .merge(right=df_team_stats, on=stg.TEAM_COL, how='left')

        return df_all_stats.fillna({col: 0 for col in df_all_stats.columns
                                    if col.endswith('_nb')})

    def _get_agg_stats(self, df_with_events, agg_column, list_events_number, list_events_with_sucess_rate):
        df_stats = df_with_events[[agg_column]].drop_duplicates()

        df_nb = self._compute_number_by_agg(df=df_with_events, agg_column=agg_column,
                                            list_event_type_id=list_events_number)
        with_sucess_rate = self._compute_number_and_sucess_rate_by_agg(df=df_with_events, agg_column=agg_column,
                                                                       list_event_type_id=list_events_with_sucess_rate)

        df_stats = df_stats.merge(right=df_nb, on=agg_column, how='left')\
                           .merge(right=with_sucess_rate, on=agg_column, how='left')

        return df_stats

    def _compute_number_by_agg(self, df, agg_column, list_event_type_id):
        NB_EVENT = '{id}_nb'

        df_number = df.query('{} in {}'.format(stg.EVENT_TYPE_COL, list_event_type_id))\
                      .groupby(by=[agg_column, stg.EVENT_TYPE_COL])\
                      .agg({stg.EVENT_TYPE_COL: 'count'})\
                      .rename(columns={stg.EVENT_TYPE_COL: 'count'})\
                      .reset_index()\
                      .pivot(index=agg_column, columns=stg.EVENT_TYPE_COL, values='count')\
                      .rename(columns={event_id: NB_EVENT.format(id=event_id)
                                       for event_id in list_event_type_id})

        return df_number

    def _compute_number_and_sucess_rate_by_agg(self, df, agg_column, list_event_type_id):
        NB_EVENT = '{id}_nb'
        SUCCESS_EVENT = '{id}_sucess'
        SUCCESS_RATE = '{id}_sucess_rate'

        df_number = self._compute_number_by_agg(df=df, agg_column=agg_column,
                                                list_event_type_id=list_event_type_id)

        df_success = df.query('{} in {}'.format(stg.EVENT_TYPE_COL, list_event_type_id))\
                       .query('{} == "1"'.format(stg.OUTCOME_COL))\
                       .groupby(by=[agg_column, stg.EVENT_TYPE_COL])\
                       .agg({stg.EVENT_TYPE_COL: 'count'})\
                       .rename(columns={stg.EVENT_TYPE_COL: 'success'})\
                       .reset_index()\
                       .pivot(index=agg_column, columns=stg.EVENT_TYPE_COL, values='success')\
                       .rename(columns={event_id: SUCCESS_EVENT.format(id=event_id)
                                        for event_id in list_event_type_id})

        df_stats = pd.merge(left=df_number, right=df_success, on=agg_column, how='left')\
                     .fillna(0)
        df_stats = df_stats.assign(**{SUCCESS_RATE.format(id=event_id): df_stats[SUCCESS_EVENT.format(id=event_id)] / df_stats[NB_EVENT.format(id=event_id)]
                                      for event_id in list_event_type_id})\
                           .drop(labels=[SUCCESS_EVENT.format(id=event_id)
                                         for event_id in list_event_type_id], axis=1)

        return df_stats

    def _fillna_game_data(self, **kwargs):
        game_data = Game(**kwargs).clean_game_data()
        game_data.fillna(value={stg.KEYPASS_COL: '0', stg.ASSIST_COL: '0'},
                         inplace=True)
        return game_data.dropna().reset_index(drop=True)

    def _get_df_major_players(self, filename=stg.FILENAME_PLAYERS_MORE_800):
        try:
            return pd.read_csv(filepath_or_buffer=join(stg.OUTPUTS_DIR, filename))
        except FileNotFoundError:
            logging.info('CSV with major players not found - Computing Players method')
            return Players().major_players


if __name__ == '__main__':
    ga = GameAnalyzer(filename='f24-24-2016-853139-eventdetails.xml')
    toto = ga._get_info_on_15_minutes(period='1', start_in_min=15,
                                      list_events_number=['4', '17'],
                                      list_events_with_sucess_rate=['1'])

    df = ga.build_game_dataset_major_players(sliding_interval_min=5,
                                             list_events_number=['4', '17'],
                                             list_events_with_sucess_rate=['1'])

    # titi = ga._compute_number_by_agg(df=toto, agg_column=stg.TEAM_COL, list_event_type_id=['4', '17'])
    # titi = ga._compute_number_by_agg(df=toto, agg_column=stg.TEAM_COL, list_event_type_id=['4', '17'])
