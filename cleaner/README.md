# TripAdvisor reviews cleaner

After scraping data from TripAdvisor, it is time to clean and tokenize the raw text of the reviews in order to use this data in an NLP algorithm.

## Architecture

The cleaner has the following subfolders:
* ``` src ``` : contains all the ```.py``` files.
* ``` assets ```: contains resources to enrich the stop words library, remove common contracted words, and add a mask to the word clouds.
* ``` cleaned_data ```: contains the tokenized reviews, word frequency dataframes (TF-IDF) and word clouds.
* ```notebooks ```: contains Jupyter notebooks to conduct Exploratory Data Analysis.

## Run from Command Line

```
python3 src/main.py --files [filenames as str] --debug --early_stop max_reviews as int
```

Usage:
* --files [str]: provides the paths to all the files to process.
* --debug: displays intermediary logs.
* --early_stop int: stops the cleaning process after the review_id reaches the given max_reviews.

## Run Exploratory Data Analysis from Jupyter Notebook

On Jupyter Notebook, execute the cells in the file ``` notebooks/EDA.ipynb ```

## Data Available in the repository

The folder ``` cleaned_data ``` contains the output from running ``` python3 src/main.py ``` to process the raw data available in the folder ``` ../scraper/scraped_data/reviews.json ```.
* The sub-folder ``` cleaned_data/restaurant_word_frequencies ``` contains files named ``` restaurant_id + _word_freq.csv ``` where the restaurant_ids are defined in the ```restaurant.json``` table from the ``` ../scraper/scraped_data ``` folder. These files contain dataframes formatted as follows:
  * as rows the review_ids, from the table ``` reviews.json ``` in the ``` ../scraper/scraped_data ``` folder, corresponding to the restaurant.
  * as columns the unique words found in all the tokenized reviews of this restaurant.
* The sub-folder ``` cleaned_data/restaurant_wordclouds ``` contains files named ``` restaurant_id + _wordcloud.png ``` where the restaurant_ids are defined in the ```restaurant.json``` table from the ``` ../scraper/scraped_data ``` folder. These are image files computed with the library ``` WordCloud ``` showing the most frequent words in the reviews of a each restaurant.
* The file ``` cleaned_data/tokenized_reviews.json ``` contains the tokenized reviews formatted as follows:
  * as key, the ```review_id``` from the table ``` reviews.json ``` from the ``` ../scraper/scraped_data ``` folder.
  * as value, a list of strings containing each tokenized word for the given review.
