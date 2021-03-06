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

### Session 4: Feature Embedder with Attention Mechanism

* Tool to embed tokenized reviews into numerical vector and predict associated ratings using a Hierarchical Attention Network (HAN).
* ``` cd attention_embedder ```
* See dedicated README in the folder.

## Run Application from Command Line

* As seen from image below simply run the following command and set user defined parameters via GUI:
```
python3 launch_program.py
```
<img width="509" alt="Screenshot 2021-03-06 at 15 41 07" src="https://user-images.githubusercontent.com/41548545/110210535-603f0580-7e92-11eb-87d7-a87537f601bf.png">

GUI User defined settings:
* --Save Wordcloud: option to create wordclouds per restaurants.
* --Save TFIDF: option to create TFIDF embedding per restaurants.
* --Embedding Technique: define embedding technique (lsi, word2vec, fasttext) supported.

Script to merge data from multiple scrapping runs, create a balanced dataset of reviews (ratings 1-5), clean selected reviews and embed words into vectors depending on user defined embedding technique (lsi, word2vec, fasttext and all are supported).

## Contributors
Project realized by @elalamik, @erraya, @hehlinge42, @louistransfer and @MaximeRedstone
