## PSG x X - Vinh Pham-Gia

### Global setup instructions and run project
This project uses Mac OS X 10.13.6 and Python 3.5.6.  
`Pipenv` package has been used to set up the project.

Therefore, one could set up the project with the following command line:
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

Environment could be activated using `make activate`.  
Once the virtual environment is activated, code could be run using the following command line:  
```
python /path_to_script/script.py
```

### Used packages
Packages used to produce such code include:
- development packages (ipdb, ipython, jedi, ipython_genutils)
- numpy
- pandas
- lxml
- sklearn
- xgboost
- tpot (with dependencies: deap, update, tqdm, stopit)
- lz4

Other models (LightGBM, Neural Networks) have been tried as well but provide worse results.

### Code architecture
#### Submission
Submitted files include mandatory files, extracts of files to manage data, saved models.

The global project could be found in the below section.

Detailed structure is as follow

    ├─ install_psgx.py    # Script to set up environment
    ├─ main_psgx.py       # Script to make predictions and write csv file
    ├─ readme.txt         # Instructions and details about the code
    ├─ ...
    ├─ code_/
    └─ settings.py

#### Global project (could be found on git repo)
The code of the project uses the Domain Driven Design architecture (DDD), which is one of the main references for projects to be industrialized and easily adaptable. Domain Driven Design includes 4 layers (infrastructure, domain, application and user interface).  
N.B: this project does not require a user interface.

Every layer corresponds to a folder with multiple python scripts and classes that interact with each other.

Structure (only relevant files are listed below)

    ├─ code_/
    │  ├─ exploration_models/               # Scripts to try models (LightGBM, Neural Networks)
    │  │  └─ ...
    │  ├─ application/
    │  │  ├─ get_best_pipeline(...).py      # Run player pipeline with different hyperparameters
    │  │  ├─ 0_main_train_all.py            # Train all pipelines on whole datasets for 3 problems
    │  │  ├─ 1_player_prediction.py         # Model and performance for player prediction
    │  │  ├─ 2_next_team_prediction.py      # Model and performance for next team prediction
    │  │  └─ 3_coordinates_predictions.py   # Model and performance for next coordinates prediction
    │  ├─ domain/
    │  │  ├─ data_processing.py             # Classes for Feature Engineering (transformers) and quality check
    │  │  ├─ games_info.py                  # Classes to extract key infos from games
    │  │  ├─ performance_analyzer.py        # Class to analyze algorithm performance
    │  │  └─ predictors.py                  # Class to manage different types of algorithms
    │  └─ infrastructure/
    │     ├─ game.py                        # Class to clean game data in XML file
    │     └─ players.py                     # Class to read players file and filter on them
    ├─ data/
    │  ├─ French-Ligue-One(...) /           # Folder with XML files of all games of first half of 2016-2017
    │  └─ Noms des joueurs et IDs(...).xml  # XML file with information about players
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




### Methodology and models
An iterative methodology has been used through the project.
Iterations have been made both on features and on model complexity.

#### Features
Problem 1 - Player prediction  
  1. Count of specific events (e.g. number of passes) by player and for corresponding team
  2. Count success rate for passes
  3. Number of goalkeeper events, number of shots
  4. Number of free kicks and corners taken

Problems 2 & 3 - Next team prediction & Coordinates prediction
  1. Iteration 1 - previous event information (previous event type, previous team, previous x coordinate)  
    - Event type is converted to float using team change rate for each event
    - X coordinate is converted to fit home team scale

#### Problem 1 - Player prediction
#### Problem 2 - Next team prediction
#### Problem 1 - Next coordinates prediction

TO BE UPDATED
