# Project Content:

## src/:
- wt_filtering.py: filters the water temperature dataset to the dates and zone specified. Depending on the nc dataset the structure can change (lat and lon order and variable names)
- upload_data.py: uploads the water temperature filtered dataset to the specified google drive account.
- functions.py:

## notebooks/:
- 01-data_collection.ipynb: collects data from Landsat and rasters the waters temperature into the desired zones
- 02-data_analysis.ipynb: Statistical distribution comparison between landsat lst data, NASA POWER air data and air stations
- 03-data_preprocessing.ipynb: Removal of unuseful satelital data and the correspondance to target data
- 04-regression_models_training.ipynb:
- 04-DL_models_training.ipynb:
- interpolaion.ipynb:
- baseline_model.ipynb:


## data/: 
Folders:
- swissTLMboiron: shapefile of swiss rivers and metadata segments
- Switzerland_shapefile: all switzerland shapefiles with different resolutions
- data: discharge and slope data about switzerland watercourse
- 30MinFreq_air_interpolated_2008_2024.csv: air temperature different stations le boiron, temporal resolution 30 min from 2011-01-01 to 2023-07-31
- water_stations_with_Qratio.csv: shp of le boiron river stations

Files:
- water_temperature.json: metadata about global water temp datasset
- waterTemperature_monthly_1981-2014.nc: global water temp dataset
 
# Instructions
## How to connect to SentinelHub:
1. Create an account
2. Go to User setttings > OAuth clients
3. Create an active OAuth client key
4. Copy the client id and the secret client id to the configuration
5. Go to Configuration Utility and create a new configuration or use an existing one
6. Copy the instance Id to the SentinelHub configuration


## Steps to upload directly to drive
1. Download the credentials file:
    - Go to the Google Developers Console.
    - Create a new project or select an existing one.
    - Enable the Google Drive API.
    - Go to "Credentials" and create a new credential of type "OAuth Client ID".
    - Download the client_secrets.json file.
2. Place the file in the correct directory:
    - Make sure you move or copy the client_secrets.json file to the same folder where your *upload_data.py* is.
3. Check the file name:
    - Make sure the file is named exactly client_secrets.json (sometimes, the operating system may add an additional extension).
4. Enable Google Drive API:
    - In the left menu, go to "APIs & Services" and then "Library."
    - Find "Google Drive API" in the library.
    - Click "Google Drive API" and then "Enable."

*Important considerations: change the folder ID according to the desired workplace.*

### How to obtain the folder ID:
1. Go to Google Drive and navigate to the folder where you want to upload the file.
2. Copy the folder ID from the URL. The ID is the part of the URL that appears after folders/