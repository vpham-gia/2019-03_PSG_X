## PSG x X - Vinh Pham-Gia

### Setup instructions to run the project
This project has been developed under Mac OS X 10.13.6 and Python 3.5.6.  
`Pipenv` package has been used through the project.

For a more convenient setup, a `requirements.txt` file is available to setup the project.  
However, `Pipfile` and `Pipfile.lock` files are provided in the folder if organizers find it more convenient.

As requested by the challenge, `install_psgx.py` allows organizers to install required packages.

Code could be therefore run using the following command lines (python 3.5 is supposed to be already installed):
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
Iterative methodology

#### Feature engineering
#### Model & pipelines
Other models (LightGBM, Neural Networks) have been tried as well but provide worse results.
