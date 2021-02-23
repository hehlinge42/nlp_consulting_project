# NLP consulting project: defining a data-driven strategy for the Londonese restaurant Bokan 37

This project has five main steps:

* Data Collection
* Data Cleaning
* Word Embedding
* Topic Extraction
* Sentiment Analysis

## Setup

```
git clone https://github.com/hehlinge42/nlp_consulting_project.git
cd nlp_consulting_project
pip install -r requirements.txt
```

## Architecture

### Session 1: TripAdvisor scraper

* Tools to scrap TripAdvisor's UK website (https://www.tripadvisor.co.uk/) for restaurants and their associated reviews made by different users.
* ``` cd scraper ```
* See dedicated README in the folder.

### Session 2: Data cleaner

* Tool to clean and tokenize the reviews scraped from TripAdvisor.
* ``` cd cleaner ```
* See dedicated README in the folder.

### Session 3: Feature Embedder

* Tool to embed tokenized reviews into numerical vector.
* ``` cd embedder ```
* See dedicated README in the folder.

## Run from Command Line

```
python3 run_all.py --embedding_technique all --re_embed False --wordcloud_per_restaurant --tfidf_per_restaurant
```

Usage:
* --embedding_technique [str]: define embedding technique (lsi, word2vec, fasttext) supported.
* --re_embed [bool]: option to rerun embedding, otherwise uses zip files provided in ./embedded_data/zip
* --wordcloud_per_restaurant [bool]: option to create wordclouds per restaurants.
* --tfidf_per_restaurant [bool]: option to create TFIDF embedding per restaurants.

Script to merge data from multiple scrapping runs, create a balanced dataset of reviews (ratings 1-5), clean selected reviews and embed words into vectors depending on user defined embedding technique (lsi, word2vec, fasttext and all are supported).

## Contributors
Project realized by @elalamik, @erraya, @hehlinge42, @louistransfer and @MaximeRedstone
