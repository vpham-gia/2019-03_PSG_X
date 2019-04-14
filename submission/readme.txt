## PSG x X - Vinh Pham-Gia
Note: a README.md file is submitted with the same content to benefit from the markdown format.

### Setup instructions to run the project
#### N.B: Feel free to get in touch if issues arise during project setup
This project has been developed under Mac OS X 10.13.6 and Python 3.5.6.
`Pipenv` package has been used through the project.

For a more convenient setup, a `requirements.txt` file is available to setup the project.
However, `Pipfile` and `Pipfile.lock` files are provided in the folder if organizers find it more convenient.

As requested by the challenge, `install_psgx.py` allows organizers to install required packages.

Code could be run using the following command lines (assuming python 3.5 is already installed):
```
python3 install_psgx.py
python3 main_psgx.py
```

Code has been tested on multiple environments:
- Ubuntu 18.04 LTS, 5 Go RAM, 1 CPU
- Mac OS High Sierra 10.13.6, 8 Go RAM, 4 CPU
- Ubuntu 18.04 LTS, 4 Go RAM, 2 CPU (lighter model is used when RAM memory is lower than 5 Go - see Model section below)

### Used packages
Packages used in the project include:
- development packages (ipdb, ipython, jedi, ipython_genutils, psutil)
- numpy
- pandas
- lxml
- sklearn
- xgboost
- tpot (with dependencies: deap, update, tqdm, stopit)
- lz4

#### Details of submitted files
Submitted files include mandatory files, saved models and simplified extracts of scripts to deal with provided data.

N.B: Files to manage games data will be released on my GitHub page at the end of the challenge. Details about global architecture will be available as well.

Detailed structure is as follow

    ├─ install_psgx.py                     # Script to set up environment
    ├─ main_psgx.py                        # Script to make predictions and write csv file
    ├─ readme.txt                          # Instructions and details about the code
    ├─ coords_x_feat (...).joblib          # Dictionary for feature engineering - project event types using average traveled X distance
    ├─ coords_x_proj_model.joblib          # Pipeline to predict next X coordinate
    ├─ coords_y_feat (...).joblib          # Dictionary for feature engineering - project event types using average traveled Y distance
    ├─ coords_y_proj_model.joblib          # Pipeline to predict next Y coordinate
    ├─ game.py                             # Class to clean game data provided in XML file
    ├─ games_info.py                       # Classes to extract key infos from games and build relevant datasets
    ├─ next_team_feat (...).joblib         # Dictionary for feature engineering - project event types using average team change
    ├─ next_team_model.joblib              # Pipeline to predict next team
    ├─ player_model_light.joblib           # Light pipeline to predict player ID (if computer RAM < 5 Go)
    ├─ player_model_missing_values.joblib  # Dictionary to impute missing values for player ID prediction
    ├─ player_model.joblib                 # Pipeline to predict player ID (if computer RAM >= 5 Go)
    ├─ README.md                           # Documentation of submission folder (markdown format, similar to readme.txt)
    ├─ requirements.txt                    # List of packages used in the project
    └─ settings.py                         # Settings used in python code (variables names, paths, characteristics of XML files, ...)

### Methodology
An iterative methodology has been used to perform required predictions.
Iterations have been made to add features and to complexify models.

#### Feature engineering
Problem 1 - Player prediction
Iterations to add more features include:
  1. Count of specific events (e.g. number of passes) by player and for corresponding team
  2. Count success rate for passes
  3. Number of goalkeeper events, number of shots
  4. Number of free kicks and corners taken
  5. Position repartition in percent

Problem 2 - Next team prediction
Iterations to add more features include:
  1. Previous event information (previous event type, previous team, previous X coordinate)
    - Event type is mapped to average percentage of team change computed by event type
    - X coordinate is converted to be consistent with home team scale
  2. Similar features are computed for previous 2 and 3 events.

Problem 3 - Next coordinates prediction
Iterations to add more features include:
  1. Previous event information (previous event type, previous team, previous X or Y coordinate)
    - Event type is mapped to average traveled distance computed by event type
    - X and Y coordinates are converted to be consistent with home team scale
  2. Similar features are computed for previous 2 and 3 events.

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
