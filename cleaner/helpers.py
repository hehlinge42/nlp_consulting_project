
import nltk
import json

import logzero
from logzero import logger
import logging
logzero.loglevel(logging.WARNING)

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

def lemmatize_and_delete_stop_words(tokenized_document, stop_words, tag_dict):
    lemmatizer = nltk.WordNetLemmatizer()
    tokens_with_tags = nltk.pos_tag(tokenized_document)
    lemmatized = []
    for token, tag in tokens_with_tags:
        if token not in stop_words:
            convert_tag = tag_dict.get(tag[0], "n")
            lemmatized.append(lemmatizer.lemmatize(token, convert_tag))
    return lemmatized, Counter(lemmatized)