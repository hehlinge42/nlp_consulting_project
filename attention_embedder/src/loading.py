import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import tqdm
import sklearn
import json
import itertools
import os
import ast 

import logzero
import logging
from logzero import logger

from helpers import gen_balanced_df
from preprocessing import review_preprocessing
from skipgram import Skipgram

BATCH_SIZE, BUFFER_SIZE = 1024, 10000

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed=42):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for vocab_size tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in dataset.
    for sequence in sequences:

        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence, 
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0
        )

        # Iterate over each positive skip-gram pair to produce training examples 
        # with positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1, 
                num_sampled=num_ns, 
                unique=True, 
                range_max=vocab_size, 
                seed=seed, 
                name="negative_sampling"
            )

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)

            context = tf.concat([context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype="int64")

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels

def get_balanced_df(filetype, filepath=None):

    save_fp = os.path.join('scraper', 'scraped_data', 'merged_data', 'balanced_dataset_' + str(filetype) + '.csv')
    logger.info(f'Looks for save_fp: {save_fp}')
    if not os.path.exists(save_fp):
        logger.info(f"Creating balanced dataset in csv as {save_fp} not found.")
        balance = False if filetype == 'gz' else True
        balanced_df = gen_balanced_df(filepath, save_fp, 'usable_rating', balance)
    else:
        logger.info(f"Reading balanced dataset csv as {save_fp} found.")
        balanced_df = pd.read_csv(save_fp, sep='#')

    logger.info("Returns balanced dataset")
    return balanced_df


def preprocess_per_model(balanced_df, tokenizer, models=['han', 'simple']):

    preprocessed_reviews_dict = {}
    logger.info(f"Preprocessing reviews for models {models}")
    if 'han' in models:
        padded_preprocessed_reviews = [review_preprocessing(review, tokenizer) for review in balanced_df["review_sentences"]]
        padded_preprocessed_reviews = tf.stack(padded_preprocessed_reviews)
        preprocessed_reviews_dict['han'] = padded_preprocessed_reviews

    if 'simple' in models:
        processed_review = [' '.join(rvw) for rvw in balanced_df["review_sentences"]]
        processed_sequences = tokenizer.texts_to_sequences(processed_review)
        preprocessed_reviews_dict['simple'] = tf.keras.preprocessing.sequence.pad_sequences(processed_sequences, maxlen=180, padding="post")        
    
    return preprocessed_reviews_dict

def get_train_test_df(balanced_df, preprocessed_reviews_dict, model_type):

    logger.info(f"Generating train and test datasets for model {model_type}")
    rating_labels = tf.keras.utils.to_categorical(balanced_df['usable_rating'], num_classes=5, dtype='float32')
    preprocessed_reviews = preprocessed_reviews_dict[model_type]

    if model_type == 'han':
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                            preprocessed_reviews.numpy(), rating_labels, 
                            test_size=0.3)
    elif model_type == 'simple':
         X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                            preprocessed_reviews, rating_labels, 
                            test_size=0.3)       

    logger.info("Split train and test")
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE)
    
    return train_ds, test_ds


def gen_sequences(balanced_df, filetype):

    review_sentences = balanced_df['review_sentences'].tolist()
    if filetype == 'json':
        review_sentences = [eval(x) for x in review_sentences]
    sentences = list(itertools.chain(*review_sentences))
    logger.info("Preprocessed sentences")

    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' ', char_level=False)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    
    logger.debug(f"{review_sentences[0][0]}")
    logger.debug(f"{sentences[0]}")
    logger.debug(f"{sequences[0]}")

    vocab_size = max(tokenizer.index_word.keys()) + 1

    return sequences, vocab_size, tokenizer


def gen_dataset(sequences, vocab_size):

    logger.info("generate_training_data")
    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=2, 
        num_ns=4, 
        vocab_size=vocab_size
    )

    logger.info("Creates dataset")
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    logger.critical(f"Dataset = {dataset}")

    return dataset


def pretrain_weights(dataset, vocab_size, embedding_dim, file_type, epochs):

    logger.info("Init Skpigram")
    word2vec = Skipgram(vocab_size=vocab_size, embedding_dim=embedding_dim)
    word2vec.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"])

    weights_dict = {}
    weight_callback = tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: weights_dict.update({epoch:word2vec.get_layer('w2v_embedding').get_weights()[0].tolist()}))

    history = word2vec.fit(dataset, epochs=epochs, callbacks=weight_callback)

    logger.info("Writing weights")
    json.dump(weights_dict, open(os.path.join("attention_embedder", "data", "weights_" + str(file_type) + ".json"),"w+"))
    logger.info("Wrote weights")
    word2vec.summary()

    pretrained_weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    
    filepath = os.path.join('.', 'attention_embedder', 'data', 'pretrained_weights_' + str(file_type) + '_' + str(vocab_size) + '.npy')
    with open(filepath, 'wb') as f:
        np.save(f, pretrained_weights)
    
    return filepath
