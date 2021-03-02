import pandas as pd
import numpy as np
import tensorflow as tf
import nltk

def stratify_data(original_df, y_column):
    min_label = original_df[y_column].value_counts().min()
    df = pd.DataFrame(columns=original_df.columns)
    for label in original_df[y_column].unique():
      subdf = original_df[original_df[y_column] == label][:min_label]
      df = df.append(subdf)
    return df

def review_preprocessing(review, words_maxlen=50, sentences_maxlen=10):
    """Preprocessing function to build appropriate padded sequences for HAN.

    Parameters
    ----------
    review: list.
        List of sentences (strings) of the review.
    
    words_maxlen: int.
        Maximal length/number of words for a sentence.

    sentences_maxlen: int.
        Maximal length/number of sentences for a review.

    Returns
    -------
    padded_sequences: tf.Tensor.
        Tensor of shape (sentences_maxlen, words_maxlen)
    """
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=' ', char_level=False)
    sequences = tokenizer.texts_to_sequences(review)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=words_maxlen, padding="post")

    if padded_sequences.shape[0] < sentences_maxlen:
        padded_sequences = tf.pad(
            padded_sequences, 
            paddings=tf.constant([[0, sentences_maxlen-padded_sequences.shape[0]], [0, 0]])
        )
    elif padded_sequences.shape[0] > sentences_maxlen:
        padded_sequences = padded_sequences[:sentences_maxlen]

    assert padded_sequences.shape == (sentences_maxlen, words_maxlen)
    return padded_sequences



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


def get_reviews(filepath, nrows=None):
    return pd.read_csv(filepath,
                     compression='gzip', 
                     low_memory=False, 
                     nrows=nrows,
                     parse_dates=['diner_date', 'rating_date'])


def clean_reviews(reviews):
    reviews['review'] = reviews.content.apply(lambda x: ' '.join(eval(x)))
    return reviews


def split_reviews_per_sentence(reviews):
    reviews["review_sentences"] = reviews.review.progress_apply(
        lambda rvw: nltk.sent_tokenize(rvw)
    )
    return reviews