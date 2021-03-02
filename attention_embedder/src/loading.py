import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import tqdm
import sklearn
import json
import itertools
import os

import logzero
import logging
from logzero import logger

from helpers import get_reviews, stratify_data, clean_reviews, split_reviews_per_sentence
from preprocessing import review_preprocessing
from skipgram import Skipgram

BATCH_SIZE, BUFFER_SIZE = 1024, 10000

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed=42):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []

    # Build the sampling table for vocab_size tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in dataset.
    for sequence in tqdm.notebook.tqdm(sequences):

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


def create_balanced_dataset(filepath, subset=1):
    
    file_type = filepath.split('.')[-1]
    logger.warn(f"Filetype is {file_type}")
    if file_type == 'gz':
        reviews = get_reviews(filepath)
        reviews = clean_reviews(reviews)
        reviews = split_reviews_per_sentence(reviews)
    elif file_type == 'json':
        with open(filepath) as json_file:
            document = json.load(json_file)
        reviews = pd.DataFrame(document)
        logger.warn(f"review has type {type(reviews)} \n {reviews.head()}")

    logger.warn(f"Review original shape = {reviews.shape}")

    if subset != 1:
        subset_length = int(subset * len(reviews))
        reviews = reviews.head(subset_length)
    logger.warn(f"Taking subset {subset} yielding shape = {reviews.shape}")
    
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

    logger.warn(f"Shape of \n X_train {X_train.shape} \n Shape of X_test {X_test.shape} \n Shape of y_train {y_train.shape} \n Shape of y_test {y_test.shape} \n")
    
    return stratified_df, train_ds, test_ds


def pretrain_weights(balanced_df, embedding_dim, file_type):
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
    
    filepath = os.path.join('.', 'attention_embedder', 'data', 'pretrained_weights_' + str(file_type) + '_' + str(vocab_size) + '.npy')
    with open(filepath, 'wb') as f:
        np.save(f, pretrained_weights)
    
    return filepath
