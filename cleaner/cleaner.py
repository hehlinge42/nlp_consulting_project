import pandas as pd
from helpers import unicode_remover, punctuation_remover, character_transformer, contraction_transformer
from helpers import delete_stop_words
from nltk.tokenize import word_tokenize
import nltk
import json

import logzero
from logzero import logger
import logging

class Cleaner():

    def __init__(self, stop_words_filename='custom_stop_words.txt'):
        self.stop_words = self.init_stop_words(stop_words_filename)


    def set_file_info(self, filename, index_col='review_id', 
                    content_col='comment', contraction_filename='contractions.json'):


        if isinstance(filename, str):
            self.filename = filename
            self.contraction_filename = contraction_filename
            json = pd.read_json(filename, lines=True)
            json.set_index(index_col, inplace = True)
            self.df = json
            self.index_col = index_col
            self.content_col = content_col
            self.tokenized_corpus = {}
        else:
            raise TypeError("Input types accepted: str")

        self.corpus = dict(zip(self.df.index, self.df[self.content_col]))
        self.init_stop_words()

    def init_stop_words(self, stop_words_filename='custom_stop_words.txt'):
        delete_from_stop_words = ['more', 'most', 'very',  'no', 'nor', 'not']
        self.stop_words = nltk.corpus.stopwords.words("english")
        self.stop_words = list(set(self.stop_words) - set(delete_from_stop_words))
        with open(stop_words_filename) as stop_words_file:
            lines = [line.rstrip() for line in stop_words_file]
        self.stop_words += lines
        logger.warn(' STOP WORDS ({})'.format(self.stop_words))
    
    def tokenize_on_steroids(self, document, ngram=1):

        if not isinstance(ngram, int):
            raise TypeError("ngram argument must be int")
        if ngram >= 1:
            tokenized_document = nltk.word_tokenize(document)
            tokenized_document = delete_stop_words(tokenized_document, self.stop_words)
            if ngram > 1:
                tokenized_document = list(nltk.ngrams(tokenized_document, n=ngram))
        else:
            raise ValueError("ngram argument must be strictly positive")
        return tokenized_document


    def clean(self):
        
        for idx, review in self.corpus.items():

            logger.warn(f' > TOKENAZING REVIEW ({idx})')

            cleaned_review = review.lower()
            cleaned_review = contraction_transformer(cleaned_review, self.contraction_filename)
            cleaned_review = character_transformer(cleaned_review)
            cleaned_review = unicode_remover(cleaned_review)
            cleaned_review = punctuation_remover(cleaned_review)
            
            tokenized_review = self.tokenize_on_steroids(cleaned_review)
            self.tokenized_corpus[idx] = tokenized_review
        
        self.tokenized_corpus

            
    def clean_new_file(self, filename, index_col='review_id', 
                    content_col='comment', contraction_filename='contractions.json'):

        self.set_file_info(filename, index_col, 
                    content_col, contraction_filename)
        self.clean()


    def write_file(self):

        with open('tokenized_reviews.json', 'w') as tokenized_reviews:
            json.dump(self.tokenized_corpus, tokenized_reviews)