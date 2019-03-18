"""Settings for the project.

Contains all configurations for the projectself.
Should NOT contain any secrets.
"""

import os
import logging

# By default the data is stored in this repository's "data/" folder.
# You can change it in your own settings file.
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_DIR = os.path.join(REPO_DIR, 'data')
GAMES_DIR = os.path.join(DATA_DIR, 'French-Ligue-One-20162017-season-Match-Day-1-19')
OUTPUTS_DIR = os.path.join(REPO_DIR, 'outputs')
MODELS_DIR = os.path.join(REPO_DIR, 'models')
LOGS_DIR = os.path.join(REPO_DIR, 'logs')


# Logging
def enable_logging(log_filename, logging_level=logging.DEBUG):
    """Enable logging."""
    with open(os.path.join(LOGS_DIR, log_filename), 'a') as file:
        file.write('\n')

    LOGGING_FORMAT = '[%(asctime)s][%(module)s] %(levelname)s - %(message)s'
    LOGGING_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATE_FORMAT,
        level=logging_level,
        filename=os.path.join(LOGS_DIR, log_filename)
    )


# Manage XML files provided by OPTA
XML_PATH_TO_GAME_INFO = '/Games/Game'
XML_PATH_TO_EVENTS = '/Games/Game/Event'
XML_PATH_TO_TEAMS = '/SoccerFeed/SoccerDocument/Team'
XML_PATH_TO_TRANSFERS = '/SoccerFeed/SoccerDocument/PlayerChanges/Team'
XML_CLUB_NAME_ATTRIBUTE = 'short_club_name'
XML_PLAYER_TAG = 'Player'
XML_LOAN_ATTRIBUTE = 'loan'
XML_PLAYER_ID_ATTRIBUTE = 'uID'
XML_PLAYER_NAME_TAG = 'Name'
XML_PLAYER_POSITION_TAG = 'Position'
XML_PLAYER_STAT_TAG = 'Stat'
XML_PLAYER_STAT_TYPE_ATTRIBUTE = 'Type'
XML_PLAYER_JOIN_DATE = 'join_date'
XML_PLAYER_LEAVE_DATE = 'leave_date'

# Game info from OPTA
PERIOD_COL = 'period_id'
EVENT_COL = 'event_id'
EVENT_TYPE_COL = 'type_id'
EVENTS_MAP = {
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

QUALIFIER_COL = 'qualifier_id'
QUALIFIER_MAP = {
    'ALL_PLAYERS': '30'
}

# Players info from OPTA
TEAM_COL = 'team_id'
PLAYER_COL = 'player_id'
PLAYER_START_COL = 'start_game'
PLAYER_EXTRA_TIME_FIRST_HALF = 'has_played_extra_time_first_half'
PLAYER_END_COL = 'end_game'
MINUTES_COL = 'min'
SECONDS_COL = 'sec'
X_COL = 'x'
X_PROJECTED_COL = 'x_along_team1_axis'
Y_COL = 'y'
Y_PROJECTED_COL = 'y_along_team1_axis'

NAME_COL = 'name'
LOAN_COL = 'loan'
TEAM_NAME_COL = 'team_name'
POSITION_COL = 'position'
ARRIVAL_DATE = 'arrival_date'
LEAVE_DATE = 'leave_date'

# Outputs filenames
FILENAME_ALL_PLAYERS = 'Noms des joueurs et IDs - F40 - L1 20162017.xml'
FILENAME_PLAYERS_MORE_800 = 'players_more_800_min.csv'
FILENAME_STATS_AGGREGATED = 'first_half_stats_by_player.csv'
FILENAME_NEXT_EVENT = 'first_half_next_event.csv'

GAME_ID_COL = 'game'
GAME_TIME_COL = 'game_time_in_sec'

COLS_TO_KEEP = [GAME_ID_COL, EVENT_TYPE_COL, PERIOD_COL, GAME_TIME_COL,
                PLAYER_COL, TEAM_COL, OUTCOME_COL, KEYPASS_COL, ASSIST_COL,
                X_COL, Y_COL]

# Model 1 - Players prediction
PLAYER_TARGET = 'player_id'
EVENTS_COMPUTE_NUMBER = ['3', '4', '7', '8', '10', '14', '16', '17', '18', '19', '44', '61']
EVENTS_COMPUTE_SUCCESS_RATE = ['1']
PLAYER_FEATURES = ['team_id']\
    + ['p_{}_nb'.format(el) for el in EVENTS_COMPUTE_SUCCESS_RATE]\
    + ['p_{}_success_rate'.format(el) for el in EVENTS_COMPUTE_SUCCESS_RATE]\
    + ['p_{}_nb'.format(el) for el in EVENTS_COMPUTE_NUMBER]\
    + ['t_{}_nb'.format(el) for el in EVENTS_COMPUTE_SUCCESS_RATE]\
    + ['t_{}_success_rate'.format(el) for el in EVENTS_COMPUTE_SUCCESS_RATE]\
    + ['t_{}_nb'.format(el) for el in EVENTS_COMPUTE_NUMBER]\

PLAYER_MODEL_NAME = 'player_model.pkl'
PLAYER_TPOT_FILENAME = 'player_tpot.py'
# ---------------
BOOL_TPOT_PLAYER = True
PLAYER_TPOT_LIMIT_TIME = 3 * 24 * 60 + 5 * 60

PLAYER_MODEL_TYPE = 'rf_classif'
BOOL_TRAIN_PLAYER_MODEL = True
BOOL_PLAYER_RS = False
# ---------------
PLAYER_TPOT_HYPERPARAMS = {'n_jobs': 3,
                           'generations': 100,
                           'population_size': 100,
                           'cv': 3,
                           'random_state': 42,
                           'verbosity': 2,
                           'max_time_mins': PLAYER_TPOT_LIMIT_TIME}
PLAYER_MODELS_HYPERPARAMS = {
    'base': {
        'rf_classif': {
            'n_estimators': 500,
            'n_jobs': 1,
            'max_depth': 15
        },
        'xgb_classif': {
            'objective': ['multi:softmax'],
            'num_class': 227,
            'n_estimators': 300,
            'learning_rate': 0.1,
            'max_depth': 6,
            'reg_lambda': 2
        }
    },
    'random_search': {
        'rf_classif': {
            'n_estimators': [50, 100, 200, 500],
            'max_features': [None, 'sqrt', 10, 15, 20],
            'max_depth': [None, 4, 8, 10, 12, 15],
            'criterion': ['gini', 'entropy']
        },
        'xgb_classif': {
            'objective': ['multi:softmax'],
            'num_class': 227,
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.8],
            'n_estimators': [30, 75, 150, 300, 500],
            'max_depth': [4, 7, 10],
            'min_child_weight': [1, 5, 10],
            'subsample': [1],
            'colsample_bytree': [0.8, 1],
            'reg_lambda': [1, 5, 10],
            'gamma': [0],
            'scale_pos_weight': [1]
        }
    }
}
PLAYER_MODEL_BASE_HYPERPARAMS = PLAYER_MODELS_HYPERPARAMS['base'][PLAYER_MODEL_TYPE]
PLAYER_RANDOM_SEARCH_HYPERPARAMS = PLAYER_MODELS_HYPERPARAMS['random_search'][PLAYER_MODEL_TYPE]

# Model 2 - Next team prediction
NEXT_EVENT_COLS_TO_LAG = [PERIOD_COL, EVENT_TYPE_COL, TEAM_COL,
                          X_PROJECTED_COL, Y_PROJECTED_COL]
NEXT_EVENT_LAGS = [1, 2, 3]

NEXT_TEAM_TARGET = TEAM_COL
NEXT_TEAM_COLS_TO_LAG_FOR_FEATS = [EVENT_TYPE_COL, TEAM_COL, X_PROJECTED_COL]
NEXT_TEAM_FEATURES = ['{}_lag{}'.format(col, lag)
                      for lag in NEXT_EVENT_LAGS
                      for col in NEXT_TEAM_COLS_TO_LAG_FOR_FEATS]

NEXT_TEAM_MODEL_NAME = 'next_team_model.pkl'
NEXT_TEAM_TPOT_FILENAME = 'next_team_tpot.py'
# ---------------
BOOL_TPOT_NEXT_TEAM = True
NEXT_TEAM_TPOT_LIMIT_TIME = 3 * 24 * 60 + 5 * 60

NEXT_TEAM_MODEL_TYPE = 'xgb_classif'
BOOL_TRAIN_NEXT_TEAM_MODEL = True
BOOL_NEXT_TEAM_RS = False
# ---------------
NEXT_TEAM_TPOT_HYPERPARAMS = {'n_jobs': 2,
                              'generations': 100,
                              'population_size': 100,
                              'cv': 3,
                              'random_state': 42,
                              'verbosity': 2,
                              'max_time_mins': NEXT_TEAM_TPOT_LIMIT_TIME}
NEXT_TEAM_MODEL_HYPERPARAMS = {
    'base': {
        'rf_classif': {'n_estimators': 500},
        'xgb_classif': {
            'n_estimators': 300,
            'learning_rate': 0.1,
            'max_depth': 6,
            'reg_lambda': 2
        }
    },
    'random_search': {
        'rf_classif': {
            'n_estimators': [50, 100, 200, 500],
            'max_features': [None, 'sqrt', 2, 6, 9],
            'max_depth': [None, 4, 8, 10, 12, 15],
            'criterion': ['gini', 'entropy']
        },
        'xgb_classif': {
            'objective': ['binary:logistic'],
            'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.8],
            'n_estimators': [30, 75, 150, 300, 500],
            'max_depth': [4, 7, 10],
            'min_child_weight': [1, 5, 10],
            'subsample': [1],
            'colsample_bytree': [0.8, 1],
            'reg_lambda': [1, 5, 10],
            'gamma': [0],
            'scale_pos_weight': [1]
        }
    }
}
NEXT_TEAM_MODEL_BASE_HYPERPARAMS = NEXT_TEAM_MODEL_HYPERPARAMS['base'][PLAYER_MODEL_TYPE]
NEXT_TEAM_RANDOM_SEARCH_HYPERPARAMS = NEXT_TEAM_MODEL_HYPERPARAMS['random_search'][PLAYER_MODEL_TYPE]

# Model 3 - Coordinates prediction
X_PROJ_TARGET = X_PROJECTED_COL
X_PROJ_COLS_TO_LAG_FOR_FEATS = [EVENT_TYPE_COL, TEAM_COL, X_PROJECTED_COL]
X_PROJ_FEATURES = ['{}_lag{}'.format(col, lag)
                   for lag in NEXT_EVENT_LAGS
                   for col in X_PROJ_COLS_TO_LAG_FOR_FEATS]
X_PROJ_MODEL_NAME = 'coords_x_proj_model.pkl'
X_PROJ_TPOT_FILENAME = 'coords_x_tpot.py'

Y_PROJ_TARGET = Y_PROJECTED_COL
Y_PROJ_COLS_TO_LAG_FOR_FEATS = [EVENT_TYPE_COL, TEAM_COL, Y_PROJECTED_COL]
Y_PROJ_FEATURES = ['{}_lag{}'.format(col, lag)
                   for lag in NEXT_EVENT_LAGS
                   for col in Y_PROJ_COLS_TO_LAG_FOR_FEATS]
Y_PROJ_MODEL_NAME = 'coords_y_proj_model.pkl'
Y_PROJ_TPOT_FILENAME = 'coords_y_tpot.py'

# ---------------
BOOL_TPOT_COORDS = True
COORDS_TPOT_LIMIT_TIME = (3 * 24 * 60 + 5 * 60) / 2

COORDS_MODEL_TYPE = 'rf_reg'
BOOL_TRAIN_COORDS_MODEL = True
BOOL_COORDS_RS = False
# ---------------
COORDS_TPOT_HYPERPARAMS = {'n_jobs': 2, 'generations': 100, 'verbosity': 2,
                           'population_size': 100, 'cv': 3, 'random_state': 42,
                           'max_time_mins': COORDS_TPOT_LIMIT_TIME}
COORDS_RF_RANDOM_SEARCH_HYPERPARAMS = {'n_estimators': [50, 100, 200, 500],
                                       'max_features': [2, 4, 6, 8],
                                       'max_depth': [4, 8, 10, 12, 15, 20, 30],
                                       'criterion': ['gini', 'entropy']}
COORDS_XGB_RANDOM_SEARCH_HYPERPARAMS = {'objective': ['reg:linear'],
                                        'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.8],
                                        'n_estimators': [30, 75, 150, 300, 500],
                                        'max_depth': [4, 7, 10],
                                        'min_child_weight': [1, 5, 10],
                                        'subsample': [1],
                                        'colsample_bytree': [0.8, 1],
                                        'reg_lambda': [1, 5, 10],
                                        'gamma': [0], 'scale_pos_weight': [1]}
X_PROJ_MODEL_HYPERPARAMS = {
    'base': {
        'rf_reg': {'n_estimators': 10},
        'xgb_reg': {'max_depth': 12, 'n_estimators': 100, 'max_features': 8}
    },
    'random_search': {
        'rf_reg': COORDS_RF_RANDOM_SEARCH_HYPERPARAMS,
        'xgb_reg': COORDS_XGB_RANDOM_SEARCH_HYPERPARAMS
    }
}
X_PROJ_MODEL_BASE_HYPERPARAMS = X_PROJ_MODEL_HYPERPARAMS['base'][COORDS_MODEL_TYPE]
X_PROJ_RANDOM_SEARCH_HYPERPARAMS = X_PROJ_MODEL_HYPERPARAMS['random_search'][COORDS_MODEL_TYPE]

Y_PROJ_MODEL_HYPERPARAMS = {
    'base': {
        'rf_reg': {'n_estimators': 10},
        'xgb_reg': {'max_depth': 12, 'n_estimators': 200, 'max_features': 6}
    },
    'random_search': {
        'rf_reg': COORDS_RF_RANDOM_SEARCH_HYPERPARAMS,
        'xgb_reg': COORDS_XGB_RANDOM_SEARCH_HYPERPARAMS
    }
}
Y_PROJ_MODEL_BASE_HYPERPARAMS = Y_PROJ_MODEL_HYPERPARAMS['base'][COORDS_MODEL_TYPE]
Y_PROJ_RANDOM_SEARCH_HYPERPARAMS = Y_PROJ_MODEL_HYPERPARAMS['random_search'][COORDS_MODEL_TYPE]

N_JOBS = 3
