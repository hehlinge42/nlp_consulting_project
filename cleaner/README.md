# TripAdvisor reviews cleaner

## Run from Command Line

```
python3 main.py
```

## Run Exploratory Data Analysis from Jupyter Notebook

On Jupyter Notebook, execute file ``` EDA.ipynb ```


## Data Available on Git

The folder ``` cleaned_data ``` contains the output computed from running ``` python3 main.py ``` on raw data available in the folder ``` ../scraper ```.
* The sub-folder ``` cleaned_data/restaurant_word_frequencies ``` contains files named ``` restaurant_ + restaurant_id + _word_freq.csv ``` according to the restaurant ids defined in the ```restaurant.json``` table from the ``` ../scraper ``` folder.
* and ``` restaurant_wordclouds ``` Data scraped on 29/01/2021 can be found in scraped_data.zip file.
It contains the data for the first 124 restaurants (maxpage_resto = 2) and their associated reviews capped at 500 per restaurant (maxpage_reviews=50).
