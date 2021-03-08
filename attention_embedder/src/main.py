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
import json
from ast import literal_eval

import matplotlib.pyplot as plt

import nltk
import sklearn
from sklearn import decomposition
import tensorflow as tf

from loading import pretrain_weights, generate_training_data, gen_dataset, gen_sequences, get_balanced_df, preprocess_per_model, get_train_test_df
from han import HierarchicalAttentionNetwork
from skipgram import Skipgram
from preprocessing import review_preprocessing
from perform_models import perform_simple_model, perform_han_model

BATCH_SIZE, BUFFER_SIZE, DATASET_SIZE = 1024, 10000, 50000

if __name__ == '__main__':

    warnings.filterwarnings('ignore')

    path_list = os.getcwd().split(os.sep)
    target_index = path_list.index('nlp_consulting_project')
    running_dir = os.path.join('.', path_list[target_index + 1])
    path_list = path_list[:target_index + 1]
    if os.name=='nt':
        path_list[0] += '/'
    os.chdir(os.path.join(os.sep, *path_list))

    parser = argparse.ArgumentParser(description='Creates Attention Embedder for Review Sentiment Classification')
    parser.add_argument('-f', '--filetype', type=str, help='csv of reviews')
    parser.add_argument('-w', '--weights', type=str, default='none', help='pretrained weights filepath')
    parser.add_argument('-m', '--model_names', nargs='*', type=str, help='name of embedding model used')
    args = parser.parse_args()

    ## Load Datasets
    if args.filetype == 'gz':
        data_fp = os.path.join('attention_embedder', 'data', 'clean_text_scrapped_data_2021.csv.gz')
    elif args.filetype == 'json':
        data_fp = os.path.join('scraper', 'scraped_data', 'merged_data', 'merged_reviews.json')
    elif args.filetype == 'csv':
        data_fp = os.path.join('attention_embedder', 'data', 'balanced_dataset_gz.csv')
    else:
        raise NotImplementedError(f"Only gz and json files are supported, found {args.filetype}")

    filepath = os.path.join('attention_embedder', 'data', args.weights)

    balanced_df = get_balanced_df(args.filetype, data_fp)
    logger.debug(f"Balance of dataset: \n{balanced_df['rating'].value_counts() / len(balanced_df)}")
    logger.debug(f"Len of dataset: {len(balanced_df)}")
    sequences, vocab_size, tokenizer = gen_sequences(balanced_df, args.filetype)
    logger.debug(f"vocab_size = {vocab_size}")
    logger.debug(f"args.model_names = {args.model_names}")
    preprocessed_reviews_dict = preprocess_per_model(balanced_df, tokenizer, args.filetype, models=args.model_names)
    # logger.critical(preprocessed_reviews_dict['han'])

    if not os.path.exists(filepath):
        logger.info(f"Weights have not been pretrained for dataset of size {balanced_df.shape}")
        dataset = gen_dataset(sequences, vocab_size)
        filepath = pretrain_weights(dataset, vocab_size, 128, file_type=args.filetype, epochs=1)
    else:
        logger.info(f"Weights have already been pretrained for dataset of size {balanced_df.shape}")


    weights = json.load(filepath)
    print(weights)
    pretrained_weights = np.array(weights["20"])
    # with open(filepath, 'r') as weights_file:
    #     logger.info(f'Loading pretrained weights from JSON file {filepath}')
    #     pretrained_weights = json.load(weights_file)
        #pretrained_weights = np.array(weights["20"])
    
    #pretrained_weights = np.load(os.path.join("..", "data", "pretrained_weights_gz_109375.npy"))
    ## Run model
    if 'simple' in args.model_names:
        logger.info(f'Running Simple Model Training')
        train_ds_simple, test_ds_simple = get_train_test_df(balanced_df, preprocessed_reviews_dict, 'simple')
        perform_simple_model(train_ds_simple, test_ds_simple, pretrained_weights)
    if 'han' in args.model_names:
        logger.info(f'Running HAN Model Training')
        train_ds_han, test_ds_han = get_train_test_df(balanced_df, preprocessed_reviews_dict, 'han')
        x = train_ds_han.take(1)
        for elem in x:
            logger.critical(elem[0])
        perform_han_model(train_ds_han, test_ds_han, pretrained_weights)