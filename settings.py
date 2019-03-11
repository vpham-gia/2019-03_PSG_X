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
    'END_PERIOD': '30',
    'DELETED_EVENT': '43'
}
OUTCOME_COL = 'outcome'
KEYPASS_COL = 'keypass'
ASSIST_COL = 'assist'

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
Y_COL = 'y'

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
PLAYER_MODEL_TYPE = 'rf'
PLAYER_MODEL_HYPERPARAMS = {'n_estimators': 500, 'n_jobs': 1}
BOOL_PLAYER_RS = True
PLAYER_RANDOM_SEARCH_HYPERPARAMS = {'n_estimators': [50, 100, 200, 500],
                                    'max_features': [None, 'sqrt', 10, 15, 20],
                                    'max_depth': [None, 4, 8, 10, 12, 15]}
# PLAYER_RANDOM_SEARCH_HYPERPARAMS = {'n_estimators': [500],
#                                     'max_features': [20],
#                                     'max_depth': [None, 15]}

# PLAYER_MODEL_HYPERPARAMS = {'n_neighbors': 20, 'n_jobs': 3}
