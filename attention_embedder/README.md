# Attention Word Embedder and Ratings Classifier

This session aims at embedding the words resulting from the tokenization of the TripAdvisor's reviews. Embedding is done using Skipgram to pretrain weights that are later used in a Hierarchical Attention Network (HAN) to predict the ratings associated to each review.


## Architecture

The attention embedder has the following subfolders:
* ``` src ``` : contains all the ```.py``` files.
* ``` data ```: contains the input data (Capgemini dataset or reviews scraped using our own TripAdvisor scraper) as well as saved pretrained weights for skipgram embedding using a balanced dataset (40k entries for Capgemini and 20k for our scraped data).
* ``` notebooks ```: contains notebook used to assess impact of hyper-parameters on HAN and evaluates predictions.
* ``` pretrained models ```: Saved best HAN models.

## Run from Command Line

From the root of ```attention_embedder``` folder:
```
python3 src/main.py --filetype gz -weights weights_folder_name.json -model_names han 
```

Usage:
* --filetype [str]: 'gz' to load Capgemini data and 'json' to load data from TripAdvisor scraper.
* --weights [str]: filename for pretrained weights file.
* --model_names [list(str)]: names of models used to predict ratings ('han' and 'simple' supported).

