# Word embedder

This session aims at embedding the words resulting from the tokenization of the TripAdvisor's reviews. Preliminary ML techniques are applied to classify the ratings in order to assess the efficiency of the different embedding techniques.


## Architecture

The embedder has the following subfolders:
* ``` src ``` : contains all the ```.py``` files.
* ``` data ```: contains the input data (clean_text_scrapped_data_2021.csv.gz for now) as well as saved pretrained weights for w2v embedding using a balanced dataset of 137 105 entries. 


## Run from Command Line

From the root of ```attention_embedder``` folder:
```
python3 src/main.py --filename
```

Usage:
* --filename [str]: filename of stored data. For now found in data folder under clean_text_scrapped_data_2021.csv.gz

