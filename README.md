## PSG x X - Vinh Pham-Gia

### Submission for the challenge
All information regarding submission setup, architecture, used packages, methodology and algorithms are further detailed in the [submission folder](submission/README.md).

### Global setup instructions and run project
This project has been developed under Mac OS X 10.13.6 and Python 3.5.6.  
`Pipenv` package has been used to set up the project.

Therefore, project setup could be performed with the following command lines:
- Install pipenv package using pip:
```
pip3 install --user pipenv
```
- Set environment variable in `.bash_profile` to create virtual environment in project folder:
```
export PIPENV_VENV_IN_PROJECT=1
```
- Set up virtual environment with all packages:
```
pipenv install
```

Code could be run using the following command line:  
```
pipenv run python /path_to_script/script.py
```

### Code architecture - Global project
The code of the project uses the Domain Driven Design architecture (DDD), which is one of the main references for projects to be industrialized and easily adaptable. Domain Driven Design includes 4 layers (infrastructure, domain, application and user interface).  
N.B: this project does not require a user interface.

Every layer corresponds to a folder with multiple python scripts and classes that interact with each other.

Structure (only relevant files are listed below)

    ├─ code_/
    │  ├─ exploration_models/               # Scripts to try models (LightGBM, Neural Networks)
    │  │  └─ ...
    │  ├─ application/
    │  │  ├─ get_best_pipeline (...).py     # Run player pipeline with different hyperparameters
    │  │  ├─ 0_main_train_all.py            # Train all pipelines on whole datasets for 3 problems
    │  │  ├─ 1_player_prediction.py         # Model and performance for player prediction
    │  │  ├─ 2_next_team_prediction.py      # Model and performance for next team prediction
    │  │  └─ 3_coordinates_predictions.py   # Model and performance for next coordinates prediction
    │  ├─ domain/
    │  │  ├─ data_processing.py             # Classes for Feature Engineering (transformers) and quality check
    │  │  ├─ games_info.py                  # Classes to extract key infos from games and build relevant datasets
    │  │  ├─ performance_analyzer.py        # Class to analyze algorithm performance
    │  │  └─ predictors.py                  # Class to manage different types of algorithms
    │  └─ infrastructure/
    │     ├─ game.py                        # Class to clean game data provided in XML file
    │     └─ players.py                     # Class to read players file and filter on them
    ├─ data/
    │  ├─ French-Ligue-One (...) /          # Folder with XML files of all games of first half of 2016-2017
    │  └─ Noms des joueurs et IDs (...).xml # XML file with information about players
    ├─ logs/                                # Folder gathering all log files
    │  └─ ...
    ├─ models/                              # Folder with saved models
    │  └─ ...
    ├─ outputs/
    │  ├─ console_logs/                     # Console logs during nohup executions
    │  │  └─ ...
    │  ├─ data/                             # Data saved
    │  │  └─ ...
    │  ├─ tpot_pipelines/                   # Output of TPOT executions
    │  │  └─ ...
    │  └─ ...
    ├─ submission/                          # See above Submission section for further details
    │  └─ ...
    ├─ ...
    ├─ Pipfile                              # Pipfile to set up environment with pipenv
    ├─ Pipfile.lock                       
    ├─ README.md
    ├─ settings_tpot.py                     # Configuration dictionaries for TPOT algorithm
    └─ settings.py                          # All settings needed in code

### Methodology
An iterative methodology has been used to perform required predictions.
Iterations have been made to add features and to complexify models.

#### Feature engineering
More details about Feature Engineering could be found in the [submission folder](submission/README.md).

#### Model & pipelines
Modeling starts from basic models and iterations tend to complexify models in order to increase performance.  
For both 3 problems, similar iterations have been made:
  1. Baseline model - dumb model (e.g. predict most frequent player, predict last team, predict last coordinates)
  2. Basic model - Random Forest with 500 trees and 15 max_depth
  3. Random search of hyperparameters for Random Forest
  4. XGBoost and random search of hyperparameters
  5. TPOT to consider more complex pipelines

Final models come from TPOT run during several days on Google Cloud Platform. Best models selected by TPOT may have been adjusted to comply with challenge requirements (files lower than 50 Mo). These adjustments have incurred lower performance.

In order to predict player ID, the best model needs a bit more than 4 Go in RAM. That is why a light model is also available, if the test computer does not satisfy this requirement.

Other models (LightGBM, Neural Networks) have been tried as well but provide worse results.
