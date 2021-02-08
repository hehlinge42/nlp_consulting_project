
import nltk
import json

from wordcloud import WordCloud
from collections import Counter

import logging
import logzero
from logzero import logger

logzero.loglevel(logging.WARNING)

def character_transformer(document):
    with_accent = ['é', 'è', 'à', "ê", "\u2019"]
    without_accent = ['e', 'e', 'a', "e", "'"]
    transformation_dict = {before:after for before, after in zip(with_accent, without_accent)}
    return document.translate(str.maketrans(transformation_dict))


def unicode_remover(document):
    encoded = document.encode('ascii', 'replace')
    return encoded.decode('utf-8')


def character_remover(document):
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


def save_wordcloud(df, idx, directory, mask):
    filename = directory + str(idx) + "_word_cloud.png"
    logger.warn(f' > WRITING {filename}')

    df_mean = df.mean().sort_values(ascending=False).to_frame(name='tfidf mean')
    dict_words_tfidf = df_mean[df_mean['tfidf mean'] != 0].to_dict()['tfidf mean']

    wordcloud = WordCloud(height=600, width=800, background_color="white",
        colormap='Blues', max_words=100, mask=mask,
        contour_width=0.5, contour_color='lightsteelblue')
    wordcloud.generate_from_frequencies(frequencies=dict_words_tfidf)
    wordcloud.to_file(filename)     


def save_tfidf(df, restaurant_id, directory, mask):
    filename = directory + str(restaurant_id) + "_word_freq.csv"
    logger.warn(f' > WRITING {filename}')

    df.to_csv(filename, index_label='review_id')