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

from helpers import get_balanced_dataset
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


def get_tf_dataset(filepath, subset=1):
    
    save_fp = os.path.join('scraper', 'scraped_data', 'merged_data', 'balanced_reviews.csv')

    if not os.path.exists(save_fp):
        logger.info(f"Creating balanced dataset in csv.")
        # reviews = preprocess(filepath)
        # logger.warn(f"Review has type {type(reviews)} \n {reviews.head()}")
        # logger.warn(f"Review original shape = {reviews.shape}")

        # if subset != 1:
        #     subset_length = int(subset * len(reviews))
        #     reviews = reviews.head(subset_length)
        # logger.warn(f"Taking subset {subset} yielding shape = {reviews.shape}")
        
        # reviews['usable_rating'] = reviews['rating'].apply(lambda r: int(r)-1)
        balanced_df = get_balanced_dataset(filepath, save_fp, 'usable_rating')
    else:
        logger.info(f"Reading balanced dataset csv.")
        balanced_df = pd.read_csv(save_fp, sep='#')

    padded_preprocessed_reviews = [review_preprocessing(review) for review in balanced_df["review_sentences"]]
    padded_preprocessed_reviews = tf.stack(padded_preprocessed_reviews)
    rating_labels = tf.keras.utils.to_categorical(balanced_df['usable_rating'], num_classes=5, dtype='float32')

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
                                padded_preprocessed_reviews.numpy(), rating_labels, 
                                test_size=0.3)


    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_ds = test_ds.batch(BATCH_SIZE)

    logger.warn(f"Shape of \n X_train {X_train.shape} \n Shape of X_test {X_test.shape} \n Shape of y_train {y_train.shape} \n Shape of y_test {y_test.shape} \n")
    
    return balanced_df, train_ds, test_ds


def pretrain_weights(balanced_df, embedding_dim, file_type, epochs):

    review_sentences = balanced_df['review_sentences'].tolist()
    review_sentences = [ast.literal_eval(x) for x in review_sentences]
    sentences = list(itertools.chain(review_sentences))

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

    word2vec.fit(dataset, epochs=epochs)
    word2vec.summary()

    pretrained_weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
    
    filepath = os.path.join('.', 'attention_embedder', 'data', 'pretrained_weights_' + str(file_type) + '_' + str(vocab_size) + '.npy')
    with open(filepath, 'wb') as f:
        np.save(f, pretrained_weights)
    
    return filepath
