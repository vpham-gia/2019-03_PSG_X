from os.path import basename, join, splitext
import logging
import os
import pandas as pd

from game import Game
import settings as stg

if __name__ == '__main__':
    stg.enable_logging(log_filename='{}.log'.format(splitext(basename(__file__))[0]),
                       logging_level=logging.DEBUG)


class NextEventInGame():
    """Class to build datasets for next event prediction for every game.

    Attributes
    ----------
    game: pandas.DataFrame
        Output of code_.infrastructure.game.Game.clean_game_data() filtered on game events

    """

    def __init__(self, **kwargs):
        """Initialize class."""
        try:
            self.game = self._filter_game_events(**kwargs)
            self.game = self._project_coordinates_along_team1_axis(df=self.game)
        except TypeError as error:
            raise NameError('Error in NextEventInGame initialization - {}'.format(error))

    def build_next_event_dataset(self, columns_to_lag, lags_to_add):
        """Build dataset with previous events.

        Attributes
        ----------
        columns_to_lag: list
            Columns to keep from self.game to use in ML
        lags_to_add: list
            Lags to perform

        Returns
        -------
        dataset: pandas.DataFrame
            DataFrame of team_id in function of previous events

        """
        try:
            assert(stg.PERIOD_COL in columns_to_lag)

            dataset = self._get_targets()

            for lag in lags_to_add:
                lag_df = self._lag_dataset(lag=lag)
                columns_name_lagged = ['{}_lag{}'.format(col, lag) for col in columns_to_lag]
                dataset = dataset.merge(right=lag_df[columns_name_lagged], how='left',
                                        right_index=True, left_index=True)\
                                 .query('{period} == {period}_lag{lag}'.format(period=stg.PERIOD_COL, lag=lag))\
                                 .drop(labels='{}_lag{}'.format(stg.PERIOD_COL, lag), axis=1)
            return dataset
        except AssertionError:
            raise KeyError('{} not in columns_to_lag list'.format(stg.PERIOD_COL))

    def _get_targets(self):
        df_target = self.game[[stg.GAME_ID_COL, stg.PERIOD_COL, stg.TEAM_COL,
                               stg.X_PROJECTED_COL, stg.Y_PROJECTED_COL]]
        return df_target

    def _lag_dataset(self, lag):
        df_lagged = self.game.shift(lag)\
                             .rename(columns={col: '{}_lag{}'.format(col, lag)
                                              for col in self.game.columns})
        return df_lagged

    def _filter_game_events(self, **kwargs):
        return Game(**kwargs)\
            .clean_game_data()\
            .reset_index(drop=True)\
            .query('{} not in {}'.format(stg.EVENT_TYPE_COL,
                                         [stg.EVENTS_MAP['START_PERIOD'], stg.EVENTS_MAP['END_PERIOD']]))

    def _project_coordinates_along_team1_axis(self, df):
        df_projections = df.assign(**{stg.X_PROJECTED_COL: lambda x: x[stg.TEAM_COL].apply(int) * x[stg.X_COL].apply(float) + (1 - x[stg.TEAM_COL].apply(int)) * (100 - x[stg.X_COL].apply(float)),
                                      stg.Y_PROJECTED_COL: lambda x: x[stg.TEAM_COL].apply(int) * x[stg.Y_COL].apply(float) + (1 - x[stg.TEAM_COL].apply(int)) * (100 - x[stg.Y_COL].apply(float))
                                      })

        return df_projections


class StatsGameAnalyzer():
    """Class to analyze a game and build datasets for ML.

    Attributes
    ----------
    game: pandas.DataFrame
        Output of code_.infrastructure.game.Game.clean_game_data()
    eligible_players: list
        Output of code_.infrastructure.players.Players.get_eligible_players_list()

    """

    def __init__(self, **kwargs):
        """Initialize class."""
        try:
            self.game = self._fillna_game_data_and_convert_types(**kwargs)
        except TypeError as error:
            raise NameError('Error in StatsGameAnalyzer initialization - {}'.format(error))

        self.eligible_players = None

    def build_stats_for_test_set(self, df_15_min, **kwargs):
        """Build dataset from cleaned events game in test set.

        Parameters
        ----------
        df_15_min: pandas.DataFrame
            Cleaned dataframe with all events within 15 minutes

        Returns
        -------
        df_stats: pandas.DataFrame
            Dataframe to make predictions on

        """
        df_players = df_15_min.query('{} == "1"'.format(stg.PLAYER_COL))\
                              .filter(items=[stg.PLAYER_COL, stg.TEAM_COL])\
                              .drop_duplicates()

        df_player_stats = self._get_agg_stats(df_with_events=df_15_min, agg_column=stg.PLAYER_COL, **kwargs)
        df_player_stats.columns = ['p_{}'.format(col) if col != stg.PLAYER_COL else col
                                   for col in df_player_stats.columns]

        df_team_stats = self._get_agg_stats(df_with_events=df_15_min, agg_column=stg.TEAM_COL, **kwargs)
        df_team_stats.columns = ['t_{}'.format(col) if col != stg.TEAM_COL else col
                                 for col in df_team_stats.columns]

        df_player_info = self._build_dataset_from_bool_columns(df_with_events=df_15_min)

        df_stats = df_players.merge(right=df_player_info, on=stg.PLAYER_COL, how='left')\
                             .merge(right=df_player_stats, on=stg.PLAYER_COL, how='left')\
                             .merge(right=df_team_stats, on=stg.TEAM_COL, how='left')
        df_stats.fillna({col: 0 for col in df_stats.columns if col.endswith('_nb')},
                        inplace=True)
        return df_stats

    def _build_dataset_from_bool_columns(self, df_with_events):
        players = pd.DataFrame({stg.PLAYER_COL: list()})

        nb_keypass = self._count_number_by_player_and_col(df=df_with_events, column=stg.KEYPASS_COL)
        nb_assist = self._count_number_by_player_and_col(df=df_with_events, column=stg.ASSIST_COL)
        nb_gk_events = self._compute_nb_gk_events(df=df_with_events)
        nb_shots = self._compute_nb_shots(df=df_with_events)
        nb_free_kick_pass = self._compute_nb_pass_after_event(df=df_with_events,
                                                              previous_event_id=stg.EVENTS_MAP['FOUL'],
                                                              colname=stg.FREE_KICK_COL)
        nb_corner = self._compute_nb_pass_after_event(df=df_with_events,
                                                      previous_event_id=stg.EVENTS_MAP['CORNER_AWARDED'],
                                                      colname=stg.CORNER_COL)

        df_out = players.merge(right=nb_keypass, on=stg.PLAYER_COL, how='outer')\
                        .merge(right=nb_assist, on=stg.PLAYER_COL, how='outer')\
                        .merge(right=nb_gk_events, on=stg.PLAYER_COL, how='outer')\
                        .merge(right=nb_shots, on=stg.PLAYER_COL, how='outer')\
                        .merge(right=nb_free_kick_pass, on=stg.PLAYER_COL, how='outer')\
                        .merge(right=nb_corner, on=stg.PLAYER_COL, how='outer')

        return df_out

    def _compute_nb_gk_events(self, df):
        df_gk_events = df.assign(**{stg.GK_EVENTS_COL: lambda x: x[stg.EVENT_TYPE_COL].apply(lambda x: 1 if x in stg.GK_EVENTS else 0)
                                    })

        df_gk_by_player = self._count_number_by_player_and_col(df=df_gk_events,
                                                               column=stg.GK_EVENTS_COL)
        return df_gk_by_player

    def _compute_nb_shots(self, df):
        df_shots = df.assign(**{stg.SHOTS_COL: lambda x: x[stg.EVENT_TYPE_COL].apply(lambda x: 1 if x in stg.SHOTS_EVENTS else 0)})

        df_shots_by_player = self._count_number_by_player_and_col(df=df_shots,
                                                                  column=stg.SHOTS_COL)
        return df_shots_by_player

    def _compute_nb_pass_after_event(self, df, previous_event_id, colname):
        PREVIOUS_EVENT = '{}_previous_event'.format(stg.EVENT_TYPE_COL)

        df_previous_event = df.shift(1)[[stg.EVENT_TYPE_COL]]\
                              .rename(columns={stg.EVENT_TYPE_COL: PREVIOUS_EVENT})

        with_prev_event = df.merge(right=df_previous_event, right_index=True,
                                   left_index=True, how='inner')

        with_prev_event['bool_event'] = with_prev_event[stg.EVENT_TYPE_COL] == stg.EVENTS_MAP['PASS']
        with_prev_event['bool_prev_event'] = with_prev_event[PREVIOUS_EVENT] == previous_event_id
        with_prev_event[colname] = with_prev_event['bool_event'] & with_prev_event['bool_prev_event']

        df_nb = self._count_number_by_player_and_col(df=with_prev_event,
                                                     column=colname)
        return df_nb

    def _count_number_by_player_and_col(self, df, column):
        df_number = df.groupby(stg.PLAYER_COL)\
                      .agg({column: 'sum'})\
                      .astype(int)\
                      .rename(columns={column: 'nb_{}'.format(column)})
        return df_number

    def _get_agg_stats(self, df_with_events, agg_column, list_events_number, list_events_with_success_rate):
        df_stats = df_with_events[[agg_column]].drop_duplicates()

        df_nb = self._compute_number_by_agg(df=df_with_events, agg_column=agg_column,
                                            list_event_type_id=list_events_number)
        with_success_rate = self._compute_number_and_success_rate_by_agg(df=df_with_events, agg_column=agg_column,
                                                                         list_event_type_id=list_events_with_success_rate)

        df_stats = df_stats.merge(right=df_nb, on=agg_column, how='left')\
                           .merge(right=with_success_rate, on=agg_column, how='left')

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

    def _compute_number_and_success_rate_by_agg(self, df, agg_column, list_event_type_id):
        NB_EVENT = '{id}_nb'
        SUCCESS_EVENT = '{id}_success'
        SUCCESS_RATE = '{id}_success_rate'

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

    def _fillna_game_data_and_convert_types(self, **kwargs):
        game_data = Game(**kwargs).clean_game_data()\
                                  .fillna(value={stg.KEYPASS_COL: '0',
                                                 stg.ASSIST_COL: '0'})\
                                  .assign(**{stg.KEYPASS_COL: lambda x: x[stg.KEYPASS_COL].apply(int),
                                             stg.ASSIST_COL: lambda x: x[stg.ASSIST_COL].apply(int)})

        return game_data.dropna().reset_index(drop=True)
