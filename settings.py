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
LOGS_DIR = os.path.join(REPO_DIR, 'logs')

# Logging
LOGGING_FORMAT = '[%(asctime)s][%(levelname)s][%(module)s] %(message)s'
LOGGING_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOGGING_LEVEL = logging.DEBUG
logging.basicConfig(
    format=LOGGING_FORMAT,
    datefmt=LOGGING_DATE_FORMAT,
    level=LOGGING_LEVEL,
    filename=os.path.join(LOGS_DIR, 'app.log')
)

XML_PATH_TO_GAME_INFO = '/Games/Game'
XML_PATH_TO_EVENTS = '/Games/Game/Event'
XML_PATH_TO_TEAMS = '/SoccerFeed/SoccerDocument/Team'
XML_CLUB_NAME_ATTRIBUTE = 'short_club_name'
XML_PLAYER_TAG = 'Player'
XML_PLAYER_ID_ATTRIBUTE = 'uID'
XML_PLAYER_NAME_TAG = 'Name'
XML_PLAYER_POSITION_TAG = 'Position'

PERIOD_COL = 'period_id'
EVENT_COL = 'event_id'
EVENT_TYPE_COL = 'type_id'
EVENTS_MAP = {
    'KICKOFF': '34',
    'PLAYER_OFF': '18',
    'PLAYER_ON': '19',
    'END_PERIOD': '30'
}

QUALIFIER_COL = 'qualifier_id'
QUALIFIER_MAP = {
    'ALL_PLAYERS': '30'
}

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
TEAM_NAME_COL = 'team_name'
POSITION_COL = 'position'

GAME_ID_COL = 'game'
GAME_TIME_COL = 'game_time_in_sec'
