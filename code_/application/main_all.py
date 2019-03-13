from os.path import abspath, join, dirname, basename, splitext
import os
import logging

import settings as stg

if __name__ == '__main__':
    stg.enable_logging(log_filename='{}.log'.format(splitext(basename(__file__))[0]),
                       logging_level=logging.DEBUG)

REPO_DIR = abspath(join(dirname(dirname(dirname(__file__)))))

logging.info('Script to predict player id ..')
os.system('python code_/application/main_player_prediction.py')
logging.info('.. Done')

logging.info('Script to predict next team ..')
os.system('python code_/application/main_next_team_prediction.py')
logging.info('.. Done')

logging.info('Script to predict coordinates ..')
os.system('python code_/application/main_coordinates_prediction.py')
logging.info('.. Done')
