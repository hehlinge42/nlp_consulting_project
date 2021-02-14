# Word embedder

This session aims at embedding the words resulting from the tokenization of the TripAdvisor's reviews. Preliminary ML techniques are applied to classify the ratings in order to assess the efficiency of the different embedding techniques.

## Architecture

The embedder has the following subfolders:
* ``` src ``` : contains all the ```.py``` files.
* ```notebooks ```: contains Jupyter notebooks to provide some visualisation and a Spark implementation of the lsi embedding in order to perform it with large datasets. 
* ``` trained_models ```: contains the ```.h5``` files to import the pretrained models without needing to run the heavy training optimization process.
* ``` embedded_data ```: contains the embedded reviews and their associated rating vectors.

## Run from Command Line

From the ```src``` folder:
```
python3 main.py --embed=True --classify=True
```

Usage:
* --embed [bool]: If set to True, the words embedding will be conducted using the lsi, word2vec and fasttext techniques. /!\ These processes are time and RAM consuming. See the **Spark implementation** section below. If set to False, the files will be read from the ```../embedded_data``` folder.
* --classify [bool]: If set to True, Deep Neural Networks will be trained in order to predict the rating of the review from its embedding version. Ignored if set to False


## Spark implementation

In order to avoid the RAM issue of conducting embedding techniques on large datasets, we implemented a Spark version of the lsi embedding. If ran on a distributed cluster, this code enables to parallelize the RAM consuming operation (TruncatedSVD) on several machines. 
This code is accessible in the ```notebooks/svd_spark.ipynb ``` notebook


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
