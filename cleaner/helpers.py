
import nltk
import json

import logzero
from logzero import logger
import logging
logzero.loglevel(logging.WARNING)

from wordcloud import WordCloud
from collections import Counter


def character_transformer(document):
    with_accent = ['é', 'è', 'à', "ê", "\u2019"]
    without_accent = ['e', 'e', 'a', "e", "'"]
    transformation_dict = {before:after for before, after in zip(with_accent, without_accent)}
    return document.translate(str.maketrans(transformation_dict))


def unicode_remover(document):
    encoded = document.encode('ascii', 'replace')
    return encoded.decode('utf-8')


def punctuation_remover(document):
    # In most of the case punctuation do not help on understanding a sentence or a doc
    characters_to_remove = ["@", "/", "#", ".", ",", "!", "?", 
                            "(", ")", "-", "_","’","'", "\"", 
                            ":", "\n", "\t", "\r"]
    transformation_dict = {initial: " " for initial in characters_to_remove}
    return document.translate(str.maketrans(transformation_dict))


def contraction_transformer(document, filename):
    with open(filename) as contractions:
        for word in document.split():
            if word in contractions:
                document = document.replace(word, contractions[word])
    return document


def lemmatize(tokenized_document, stop_words, tag_dict):
    lemmatizer = nltk.WordNetLemmatizer()
    tokens_with_tags = nltk.pos_tag(tokenized_document)
    lemmatized = []
    for token, tag in tokens_with_tags:
        if token not in stop_words:
            convert_tag = tag_dict.get(tag[0], "n")
            lemmatized.append(lemmatizer.lemmatize(token, convert_tag))
    return lemmatized


def save_wordcloud(df, filename, idx, directory, mask):

    filename = directory + filename + str(idx) + "_word_cloud.png"
    df_mean = df.mean().sort_values(ascending=False).to_frame(name='tfidf mean')
    dict_words_tfidf = df_mean[df_mean['tfidf mean'] != 0].to_dict()['tfidf mean']

    wordcloud = WordCloud(height=600, width=800, background_color="white",
        colormap='Blues', max_words=100, mask=mask,
        contour_width=0.5, contour_color='lightsteelblue')
    wordcloud.generate_from_frequencies(frequencies=dict_words_tfidf)
    wordcloud.to_file(filename)     

    logger.warn(f' > WRITING {filename}')


def save_tfidf(df, filename, restaurant_id, directory, mask):
    filename = directory + filename + str(restaurant_id) + "_word_freq.csv"
    df.to_csv(filename)
    logger.warn(f' > WRITING {filename}')