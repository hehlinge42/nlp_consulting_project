import os
import io
import re
import argparse

import logging
import logzero
from logzero import logger

import pandas as pd
import numpy as np

import tqdm
import warnings
import itertools
from ast import literal_eval

# import seaborn as sns
import matplotlib.pyplot as plt
# import plotly.express as px

import nltk
import sklearn
from sklearn import decomposition
import tensorflow as tf

from loading import create_balanced_dataset, pretrain_weights, generate_training_data
from han import HierarchicalAttentionNetwork
from skipgram import Skipgram
from preprocessing import review_preprocessing

BATCH_SIZE, BUFFER_SIZE, DATASET_SIZE = 1024, 10000, 50000

if __name__ == '__main__':

    ## Setup environment ##
    warnings.filterwarnings('ignore')
    tqdm.tqdm_notebook()
    tqdm.notebook.tqdm().pandas()

    path_list = os.getcwd().split(os.sep)
    target_index = path_list.index('nlp_consulting_project')
    running_dir = os.path.join('.', path_list[target_index + 1])
    path_list = path_list[:target_index + 1]
    os.chdir(os.path.join(os.sep, *path_list))

    parser = argparse.ArgumentParser(description='Creates Attention Embedder for Review Sentiment Classification')
    parser.add_argument('-f', '--filename', type=str, help='csv of reviews')
    parser.add_argument('-m', '--model_name', type=str, help='name of embedding model saved after training')
    args = parser.parse_args()

    ## Load Datasets
    logger.warn(f"Loading balanced dataset from {args.filename}")
    file_type = args.filename.split('.')[-1]
    if file_type == 'gz':
        data_fp = os.path.join('.', 'attention_embedder', 'data', 'clean_text_scrapped_data_2021.csv.gz')
    elif file_type == 'json':
        data_fp = os.path.join('.', 'scraper', 'scraped_data', 'merged_data', 'merged_reviews.json')
    else:
        raise NotImplementedError(f"Only gz and json files are supported, found {file_type}")

    balanced_df, train_ds, test_ds = create_balanced_dataset(data_fp, 0.05)
    
    filepath = os.path.join('.', 'attention_embedder', 'data', 'pretrained_weights_38390.npy')
    if not os.path.exists(filepath):
        logger.warn("Weights have not been pretrained.")
        filepath = pretrain_weights(balanced_df, 128, file_type=args.filename.split('.')[-1])

    pretrained_weights = np.load(filepath)
    vocab_size = int(filepath.split('_')[-1].split('.')[0])
    logger.warn(f"Vocab size = {vocab_size}")

    ## Run model
    han_model_reg = HierarchicalAttentionNetwork(vocab_size=vocab_size, embedding_dim=128, 
                    pretrained_weights=pretrained_weights, gru_units=32, attention_units=32, 
                    classifier_units=5, dropout_embedding=0.2, recurrent_dropout=0.2, 
                    callable=tf.keras.regularizers.l2, penalty=1e-05)

    han_model_reg.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           optimizer="adam", metrics=["accuracy"])

    han_history_reg = han_model_reg.fit(train_ds, epochs=20, validation_data=test_ds)
    han_model_reg.save(os.path.join('.', 'attention_embedder', 'pretrained_models', args.model_name))