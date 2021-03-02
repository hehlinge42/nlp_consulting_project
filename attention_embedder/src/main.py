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

from loading import *
from han import HierarchicalAttentionNetwork
from skipgram import Skipgram

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
DATASET_SIZE = 50000

## Setup environment
warnings.filterwarnings('ignore')
tqdm.tqdm_notebook()
tqdm.notebook.tqdm().pandas()

def create_balanced_dataset(filepath):
    
    reviews = get_reviews(filepath, DATASET_SIZE)
    reviews = clean_reviews(reviews)
    reviews = split_reviews_per_sentence(reviews)
    logger.warn(f"Review shape = {reviews.shape}")

    reviews['usable_rating'] = reviews['rating'].apply(lambda r: int(r)-1)
    stratified_df = stratify_data(reviews, 'usable_rating')
    padded_preprocessed_reviews = [review_preprocessing(review) for review in tqdm.notebook.tqdm(stratified_df["review_sentences"])]
    padded_preprocessed_reviews = tf.stack(padded_preprocessed_reviews)
    rating_labels = tf.keras.utils.to_categorical(stratified_df['usable_rating'], num_classes=5, dtype='float32')

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                                padded_preprocessed_reviews.numpy(), rating_labels, 
                                test_size=0.3)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE)

    logger.warn(f"Shape of X_train {X_train.shape} \n Shape of X_test {X_test.shape} \n Shape of y_train {y_train.shape} \n Shape of y_test {y_test.shape} \n")
    return train_ds, test_ds

def pretrain_weights(balanced_df, embedding_dim):
    sentences = list(itertools.chain(*balanced_df["review_sentences"]))
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' ', char_level=False)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    vocab_size = max(tokenizer.index_word.keys()) + 1

    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=2, 
        num_ns=4, 
        vocab_size=vocab_size
    )

    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    word2vec = Skipgram(vocab_size=max(tokenizer.index_word.keys())+1, embedding_dim=embedding_dim)
    word2vec.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])

    word2vec.fit(dataset, epochs=1)
    word2vec.summary()

    pretrained_weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    
    filepath = os.path.join('..', 'data', 'pretrained_weights_' + str(vocab_size) + '.npy')
    with open(filepath, 'wb') as f:
        np.save(f, pretrained_weights)
    
    return filepath

if __name__ == '__main__':

    path_list = os.getcwd().split(os.sep)
    target_index = path_list.index('nlp_consulting_project')
    running_dir = os.path.join('.', path_list[target_index + 1])
    path_list = path_list[:target_index + 1]
    os.chdir(os.path.join(os.sep, *path_list))

    parser = argparse.ArgumentParser(description='Creates Attention Embedder for Review Sentiment Classification')
    parser.add_argument('-f', '--filename', type=str, default=False, help='csv of reviews')
    args = parser.parse_args()

    # if first run
    logger.warn(f"Loading balanced dataset from {args.filename}")
    train_ds, test_ds = create_balanced_dataset(os.path.join('.', 'attention_embedder', 'data', args.filename))
    filepath = ' '
    if not os.path.exists(filepath):
        logger.warn("Weights have not been pretrained.")
        filepath = pretrain_weights(train_ds, 128)

    pretrain_weights = np.load(filepath)
    vocab_size = int(filename.split('_')[:-1].split('.')[0])
    logger.warn(f"vocab size = {vocab_size}")


    ## Run model
    han_model_reg = HierarchicalAttentionNetwork(vocab_size=vocab_size, embedding_dim=128, 
                    pretrained_weights=pretrained_weights, gru_units=32, attention_units=32, 
                    classifier_units=5, dropout_embedding=0.2, recurrent_dropout=0.2, 
                    callable=tf.keras.regularizers.l2,
                    penalty=1e-05)

    han_model_reg.cosmpile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                           optimizer="adam", metrics=["accuracy"])

    han_history_reg = han_model_reg.fit(train_ds, epochs=20, validation_data=test_ds)