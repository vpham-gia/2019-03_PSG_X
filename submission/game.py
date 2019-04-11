import pandas as pd

from lxml import etree

import settings as stg
import logging


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
        for col in list_cols:
            if col not in df.columns:
                logging.debug('File {}: {} not in columns'.format(self.filename, col))
                df[col] = '0'
        return df[[col for col in list_cols if col in df.columns]]

    def _gather_events(self):
        tree = etree.parse(self.filename)

        df_events = pd.DataFrame()

        for event in tree.xpath(stg.XML_PATH_TO_EVENTS):
            df_event = pd.DataFrame(dict(event.attrib), index=[0])
            # df_qualifiers = self._get_qualifiers_for_event(event_xml_element=event)
            # with_qualifiers = pd.merge(left=df_event, right=df_qualifiers,
            #                            left_index=True, right_index=True, how='left')

            df_events = pd.concat([df_events, df_event],
                                  axis=0, ignore_index=True, sort=False)

        return df_events

    def _get_qualifiers_for_event(self, event_xml_element):
        qualifiers = event_xml_element.findall(stg.XML_QUALIFIER_TAG)
        # To filter on qualifier 56
        # qualifiers = event_xml_element.findall('{}/[@{}="56"]'.format(stg.XML_QUALIFIER_TAG, stg.QUALIFIER_COL))

        df_qualifiers = pd.DataFrame({
            stg.QUALIFIER_COL: list(map(lambda x: x.get(stg.QUALIFIER_COL), qualifiers)),
            stg.XML_QUALIFIER_VALUE_ATTRIBUTE: list(map(lambda x: x.get(stg.XML_QUALIFIER_VALUE_ATTRIBUTE), qualifiers)),
        })

        df_in_row = df_qualifiers.set_index(stg.QUALIFIER_COL)\
                                 .transpose()\
                                 .reset_index(drop=True)\
                                 .fillna('value')
        df_in_row.columns = ['{}_{}'.format(stg.QUALIFIER_COL, col)
                             for col in df_in_row.columns]

        return df_in_row

    def _get_home_and_away_id(self):
        tree = etree.parse(self.filename)

        game_info = tree.xpath(stg.XML_PATH_TO_GAME_INFO)[0]
        return game_info.get('home_team_id'), game_info.get('away_team_id')
