from os.path import basename, join, splitext
import logging
import os
import pandas as pd

from code_.infrastructure.game import Game
from code_.infrastructure.players import Players
import settings as stg

if __name__ == '__main__':
    stg.enable_logging(log_filename='{}.log'.format(splitext(basename(__file__))[0]),
                       logging_level=logging.DEBUG)


class SeasonFirstHalfAggregator():
    """Class to aggregate all stats from first half of season.

    Attributes
    ----------
    sliding_interval_min: int
        Time delta to slide time window in every game
    list_events_number: list
        Events type_id for which number aggregation are computed
    list_events_with_success_rate: list
        Events type_id for which number and success rate are computed
    saved_filename: string, default FILENAME_STATS_AGGREGATED in settings

    """

    def __init__(self, sliding_interval_min, list_events_number,
                 list_events_with_success_rate,
                 saved_filename=stg.FILENAME_STATS_AGGREGATED):
        """Class init."""
        self.sliding_interval_min = sliding_interval_min
        self.list_events_number = list_events_number
        self.list_events_with_success_rate = list_events_with_success_rate

        self.saved_filename = saved_filename

    def build_dataset(self):
        """Aggregate stats info for all games in first half of season.

        Returns
        -------
        df_stats: pandas.DataFrame
            Aggregation of stats for all games

        """
        try:
            return pd.read_csv(join(stg.OUTPUTS_DIR, self.saved_filename))
        except FileNotFoundError:
            logging.debug('Start to aggregate game stats..')
            number_games_processed = 0

            df_stats = pd.DataFrame()

            for file_ in os.listdir(stg.GAMES_DIR):
                stats_game = StatsGameAnalyzer(filename=file_)\
                    .build_game_dataset_eligible_players(sliding_interval_min=self.sliding_interval_min,
                                                         list_events_number=self.list_events_number,
                                                         list_events_with_success_rate=self.list_events_with_success_rate)

                df_stats = pd.concat([df_stats, stats_game], axis=0,
                                     ignore_index=True, sort=False)
                number_games_processed += 1
                logging.debug('.. {}/{} - successfully loaded file {}'
                              .format(number_games_processed, len(os.listdir(stg.GAMES_DIR)), file_))

            df_stats.fillna({col: 0 for col in df_stats.columns if col.endswith('_nb')})\
                    .to_csv(join(stg.OUTPUTS_DIR, self.saved_filename))
            logging.info('Successfully saved aggregated stats.')
            return df_stats

    def build_next_team_dataset(self, columns_to_lag, filename=stg.FILENAME_NEXT_TEAM):
        """Aggregate team with previous events for all games in first half of season.

        Returns
        -------
        df_next_team: pandas.DataFrame
            Aggregation teams with previous events

        """
        try:
            return pd.read_csv(join(stg.OUTPUTS_DIR, filename))
        except FileNotFoundError:
            logging.debug('Start to aggregate next team dataset..')
            number_games_processed = 0

            df_next_team = pd.DataFrame()

            for file_ in os.listdir(stg.GAMES_DIR):
                game_next_team = NextEventInGame(filename=file_)\
                    .build_next_team_dataset(columns_to_lag=columns_to_lag)

                df_next_team = pd.concat([df_next_team, game_next_team], axis=0,
                                         ignore_index=True, sort=False)
                number_games_processed += 1
                logging.debug('.. {}/{} - successfully loaded file {}'
                              .format(number_games_processed, len(os.listdir(stg.GAMES_DIR)), file_))

            df_next_team.to_csv(join(stg.OUTPUTS_DIR, filename))
            logging.info('Successfully saved aggregated next team dataset.')
            return df_next_team


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
        except TypeError as error:
            raise NameError('Error in NextEventInGame initialization - {}'.format(error))

    def build_next_team_dataset(self, columns_to_lag):
        """Build dataset with previous events.

        Attributes
        ----------
        columns_to_lag: list
            Columns to keep from self.game to use in ML

        Returns
        -------
        dataset: pandas.DataFrame
            DataFrame of team_id in function of previous events

        """
        try:
            assert(stg.PERIOD_COL in columns_to_lag)

            target = self._get_target(target_col=stg.TEAM_COL)

            lag = 1
            lag_df = self._lag_dataset(lag=1)
            columns_name_lagged = ['{}_lag{}'.format(col, lag) for col in columns_to_lag]
            dataset = target.merge(right=lag_df[columns_name_lagged], how='left',
                                   right_index=True, left_index=True)\
                            .query('{period} == {period}_lag{lag}'.format(period=stg.PERIOD_COL, lag=lag))\
                            .drop(labels='{}_lag{}'.format(stg.PERIOD_COL, lag), axis=1)
            return dataset
        except AssertionError:
            raise KeyError('{} not in columns_to_lag list'.format(stg.PERIOD_COL))

    def _get_target(self, target_col):
        df_target = self.game[[stg.GAME_ID_COL, stg.PERIOD_COL, target_col]]
        return df_target

    def _lag_dataset(self, lag):
        # TODO: add variable creation in other method
        df_x_projected = self.game.assign(**{stg.X_PROJECTED_COL:
                                             lambda x: x[stg.TEAM_COL].apply(int) * x[stg.X_COL].apply(float) + (1 - x[stg.TEAM_COL].apply(int)) * (100 - x[stg.X_COL].apply(float))})
        df_lagged = df_x_projected.shift(lag)\
                                  .rename(columns={col: '{}_lag{}'.format(col, lag)
                                                   for col in df_x_projected.columns})

        return df_lagged

    def _filter_game_events(self, **kwargs):
        return Game(**kwargs)\
            .clean_game_data()\
            .reset_index(drop=True)\
            .query('{} not in {}'.format(stg.EVENT_TYPE_COL,
                                         [stg.EVENTS_MAP['START_PERIOD'], stg.EVENTS_MAP['END_PERIOD']]))


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
            self.game = self._fillna_game_data(**kwargs)
        except TypeError as error:
            raise NameError('Error in StatsGameAnalyzer initialization - {}'.format(error))

        self.eligible_players = Players().get_eligible_players_list()

    def build_game_dataset_eligible_players(self, sliding_interval_min,
                                            list_events_number,
                                            list_events_with_success_rate):
        """Compute stats for every period with 15 min intervals for major players.

        Parameters
        ----------
        sliding_interval_min: integer
            Time interval to move sliding window
        list_events_number: list
            Event types ID for which number of events are computed
        list_events_with_success_rate: list
            Event types ID for which number of events and success rate are computed

        Returns
        -------
        df_game_stats: pandas.DataFrame
            Columns: game, player_id, binarized team_id, time_interval, stats

        """
        df_game_stats = pd.DataFrame()

        for start in range(0, 36, sliding_interval_min):
            df = self._get_info_on_15_minutes(period='1', start_in_min=start,
                                              list_events_number=list_events_number,
                                              list_events_with_success_rate=list_events_with_success_rate)\
                     .query('{} in {}'.format(stg.PLAYER_COL, self.eligible_players))
            df_game_stats = pd.concat([df_game_stats, df], axis=0,
                                      ignore_index=True, sort=False)

        for start in range(45, 76, sliding_interval_min):
            df = self._get_info_on_15_minutes(period='2', start_in_min=start,
                                              list_events_number=list_events_number,
                                              list_events_with_success_rate=list_events_with_success_rate)\
                     .query('{} in {}'.format(stg.PLAYER_COL, self.eligible_players))
            df_game_stats = pd.concat([df_game_stats, df], axis=0,
                                      ignore_index=True, sort=False)

        df_game_stats.fillna({col: 0 for col in df_game_stats.columns if col.endswith('_nb')},
                             inplace=True)

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
        return df_all_stats

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

    def _fillna_game_data(self, **kwargs):
        game_data = Game(**kwargs).clean_game_data()
        game_data.fillna(value={stg.KEYPASS_COL: '0', stg.ASSIST_COL: '0'},
                         inplace=True)
        return game_data.dropna().reset_index(drop=True)


if __name__ == '__main__':
    # ga = StatsGameAnalyzer(filename='f24-24-2016-853139-eventdetails.xml')
    # pl = Players()
    #
    # df = ga.build_game_dataset_eligible_players(sliding_interval_min=5,
    #                                             list_events_number=['4', '17'],
    #                                             list_events_with_success_rate=['1'])
    #
    # df = df.merge(right=pl.all_players, on='player_id', how='left')

    # neig = NextEventInGame(filename='f24-24-2016-853139-eventdetails.xml')
    # df = neig.build_next_team_dataset(columns_to_lag=[stg.PERIOD_COL] + stg.NEXT_TEAM_COLS_TO_LAG)

    sfha = SeasonFirstHalfAggregator(sliding_interval_min=5, list_events_number=[], list_events_with_success_rate=[])
    df = sfha.build_next_team_dataset(columns_to_lag=stg.NEXT_TEAM_COLS_TO_LAG)
