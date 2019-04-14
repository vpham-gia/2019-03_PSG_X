"""Settings for the project.

Contains all configurations for the projectself.
Should NOT contain any secrets.
"""

import logging


# Logging
def enable_logging(log_filename, logging_level=logging.DEBUG):
    """Enable logging."""
    with open(log_filename, 'a') as file:
        file.write('\n')

    LOGGING_FORMAT = '[%(asctime)s][%(module)s] %(levelname)s - %(message)s'
    LOGGING_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATE_FORMAT,
        level=logging_level,
        filename=log_filename
    )


# Manage XML files provided by OPTA
XML_PATH_TO_GAME_INFO = '/Games/Game'
XML_PATH_TO_EVENTS = '/Games/Game/Event'
XML_QUALIFIER_TAG = 'Q'
XML_QUALIFIER_VALUE_ATTRIBUTE = 'value'

# Game info from OPTA
PERIOD_COL = 'period_id'
EVENT_COL = 'event_id'
EVENT_TYPE_COL = 'type_id'
EVENTS_MAP = {
    'PASS': '1',
    'FOUL': '4',
    'CORNER_AWARDED': '6',
    'KICKOFF': '34',
    'PLAYER_OFF': '18',
    'PLAYER_ON': '19',
    'START_PERIOD': '32',
    'END_PERIOD': '30',
    'DELETED_EVENT': '43'
}
OUTCOME_COL = 'outcome'
KEYPASS_COL = 'keypass'
ASSIST_COL = 'assist'

GK_EVENTS = ['10', '11', '41', '52', '53', '54', '58']
GK_EVENTS_COL = 'gk_events'
SHOTS_EVENTS = ['13', '14', '15', '16']
SHOTS_COL = 'shots'
FREE_KICK_COL = 'free_kick'
CORNER_COL = 'corner'

QUALIFIER_COL = 'qualifier_id'
QUALIFIER_MAP = {
    'ALL_PLAYERS': '30',
    'ZONE': '56'
}

# Players info from OPTA
TEAM_COL = 'team_id'
PLAYER_COL = 'player_id'
MINUTES_COL = 'min'
SECONDS_COL = 'sec'
X_COL = 'x'
X_PROJECTED_COL = 'x_along_team1_axis'
Y_COL = 'y'
Y_PROJECTED_COL = 'y_along_team1_axis'

GAME_ID_COL = 'game'
GAME_TIME_COL = 'game_time_in_sec'

COLS_TO_KEEP = [GAME_ID_COL, EVENT_TYPE_COL, '{}_{}'.format(QUALIFIER_COL, QUALIFIER_MAP['ZONE']),
                PERIOD_COL, GAME_TIME_COL, PLAYER_COL, TEAM_COL, OUTCOME_COL, KEYPASS_COL, ASSIST_COL,
                X_COL, Y_COL]

# Model 1 - Players prediction
EVENTS_COMPUTE_NUMBER = ['3', '4', '7', '8', '10', '14', '16', '17', '18', '19', '44', '61']
EVENTS_COMPUTE_SUCCESS_RATE = ['1']
PLAYER_FEATURES = ['team_id']\
    + ['nb_assist', 'nb_keypass', 'nb_gk_events', 'nb_shots', 'nb_free_kick', 'nb_corner']\
    + ['Back', 'Center', 'Left', 'Right']\
    + ['p_{}_nb'.format(el) for el in EVENTS_COMPUTE_SUCCESS_RATE]\
    + ['p_{}_success_rate'.format(el) for el in EVENTS_COMPUTE_SUCCESS_RATE]\
    + ['p_{}_nb'.format(el) for el in EVENTS_COMPUTE_NUMBER]\
    + ['t_{}_nb'.format(el) for el in EVENTS_COMPUTE_SUCCESS_RATE]\
    + ['t_{}_success_rate'.format(el) for el in EVENTS_COMPUTE_SUCCESS_RATE]\
    + ['t_{}_nb'.format(el) for el in EVENTS_COMPUTE_NUMBER]\

PLAYER_FEATURES_MEDIAN_FILENAME = 'player_model_missing_values.joblib'

PLAYER_MODEL_NAME = 'player_model.joblib'
PLAYER_MODEL_LIGHT_NAME = 'player_model_light.joblib'

# Model 2 - Next team prediction
NEXT_EVENT_COLS_TO_LAG = [PERIOD_COL, EVENT_TYPE_COL, TEAM_COL,
                          X_PROJECTED_COL, Y_PROJECTED_COL]
NEXT_EVENT_LAGS = [1, 2, 3]

NEXT_TEAM_COLS_TO_LAG_FOR_FEATS = [EVENT_TYPE_COL, TEAM_COL, X_PROJECTED_COL]
NEXT_TEAM_FEATURES = ['{}_lag{}'.format(col, lag)
                      for lag in NEXT_EVENT_LAGS
                      for col in NEXT_TEAM_COLS_TO_LAG_FOR_FEATS]

NEXT_TEAM_FEAT_ENG_DICT = 'next_team_feature_engineering_dict.joblib'
NEXT_TEAM_MODEL_NAME = 'next_team_model.joblib'

# Model 3 - Coordinates prediction
X_PROJ_COLS_TO_LAG_FOR_FEATS = [EVENT_TYPE_COL, TEAM_COL, X_PROJECTED_COL]
X_PROJ_FEATURES = ['{}_lag{}'.format(col, lag)
                   for lag in NEXT_EVENT_LAGS
                   for col in X_PROJ_COLS_TO_LAG_FOR_FEATS]

X_PROJ_FEAT_ENG_DICT = 'coords_x_feature_engineering_dict.joblib'
X_PROJ_MODEL_NAME = 'coords_x_proj_model.joblib'

Y_PROJ_COLS_TO_LAG_FOR_FEATS = [EVENT_TYPE_COL, TEAM_COL, Y_PROJECTED_COL]
Y_PROJ_FEATURES = ['{}_lag{}'.format(col, lag)
                   for lag in NEXT_EVENT_LAGS
                   for col in Y_PROJ_COLS_TO_LAG_FOR_FEATS]

Y_PROJ_FEAT_ENG_DICT = 'coords_y_feature_engineering_dict.joblib'
Y_PROJ_MODEL_NAME = 'coords_y_proj_model.joblib'
