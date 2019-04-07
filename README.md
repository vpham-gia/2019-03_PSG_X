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

### Code architecture
#### Global project (could be found on git repo)
The code of the project uses the Domain Driven Design architecture (DDD), which is one of the main references for projects to be industrialized and easily adaptable. Domain Driven Design includes 4 layers (infrastructure, domain, application and user interface).

Each layer relies on the previous ones and cannot use higher layers.
Every layer is actually a folder with multiple python scripts and classes that interact with each other.

Structure

    ├─ matrix/
    │  ├─ application/
    │  │  └─ gather_all_schemas.py            # Script to collect all schemas from db
    │  ├─ domain/
    │  │  └─ geopandas_convertor.py           # Script to convert geometric elements to dataframe
    │  ├─ infrastructure/
    │  │  └─ postgre_connector.py             # Script to connect to PostgreSQL db
    │  └─ interface/
    │     ├─ docs/                            # Documentation to manage dash components
    │     ├─ pages/                           # Pages for every available tab - each page includes front-end part with elements to display and callbacks to manage interactions between those elements
    │     │  ├─ communes.py
    │     │  ├─ graphiques.py
    │     │  ├─ home.py
    │     │  ├─ map_nro.py
    │     │  ├─ map.py
    │     │  ├─ reports_queries.yaml          # SQL queries used in reports.py to get the data from db
    │     │  ├─ reports.py                    # Displays reports buttons to download main KPI - this tab uses SQL queries in reports_queries.yaml
    │     │  └─ scenarios.py
    │     ├─ partials/
    │     │  ├─ header.py                     # DO NOT EDIT - Header
    │     │  └─ menu.py                       # Side menu to add or remove tab
    │     ├─ static/                          # DO NOT EDIT - Static elements including images, JS scripts and CSS stylesheets
    │     ├─ app_spa.py                       # DO NOT EDIT - Example of single-page application
    │     ├─ app.py                           # DO NOT EDIT - Main application
    │     ├─ index.py                         # Main python file
    │     ├─ README.md
    │     ├─ router.py                        # Routing to pages
    │     └─ static.py                        # DO NOT EDIT
    ├─ settings/                              # Settings for matrix project
    │  ├─ .env                                # Contains secret environment variables such as DATABASE_PASSWORD and MAPBOX_TOKEN
    │  ├─ .env_template                       # .env template to be duplicated to create .env file
    │  └─ base.py                             # Public variables to be used through all project
    ├─ tutorials/                             # Basic documentation about Git
    ├─ gcp_init.sh                            # INIT - Bash command lines to initialize project
    ├─ README.md
    └─ requirements.txt                       # INIT - Contains python packages to be installed
#### Submission

### Methodology
#### Problem 1 - Player prediction
#### Problem 2 - Next team prediction
#### Problem 1 - Next coordinates prediction

TO BE UPDATED
