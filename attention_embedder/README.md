# Attention Word Embedder and Ratings Classifier

This session aims at embedding the words resulting from the tokenization of the TripAdvisor's reviews. Embedding is done using Skipgram to pretrain weights that are later used in a Hierarchical Attention Network (HAN) to predict the ratings associated to each review.


## Architecture

The attention embedder has the following subfolders:
* ``` src ``` : contains all the ```.py``` files.
* ``` data ```: contains the input data (Capgemini dataset or reviews scraped using our own TripAdvisor scraper) as well as saved pretrained weights for skipgram embedding using a balanced dataset (40k entries for Capgemini and 20k for our scraped data).
* ``` notebooks ```: contains notebook used to assess impact of hyper-parameters on HAN and evaluates predictions.
* ``` pretrained models ```: Saved best HAN models.

## Run Notebook

From the root of ```attention_embedder/notebooks``` folder, import the ```attention_embedder_regularized.ipynb``` notebook in Google Colab and run all.
