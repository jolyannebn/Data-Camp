# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import unicode_literals
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
import unicodedata
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import string
from urllib.parse import urlparse
import urllib.request


def findTitle(url):
    try: 
        webpage = urllib.request.urlopen(url).read().decode('utf-8')
        title = str(webpage).split('<title>')[1].split('</title>')[0]
        return title
    except urllib.error.URLError as e:
        #print(e.reason, url)
        return url

def document_preprocessor(doc):
    """ A custom document preprocessor

    This function can be edited to add some additional
    transformation on the documents prior to tokenization.

    At present, this function passes the document through
    without modification.
    """
    return doc

def clean_str(sentence, stem=True):
    english_stopwords = set(
        [stopword for stopword in stopwords.words('english')])
    punctuation = set(string.punctuation)
    punctuation.update(["``", "`", "..."])
    if stem:
        stemmer = SnowballStemmer('english')
        return list((filter(lambda x: x.lower() not in english_stopwords and
                            x.lower() not in punctuation,
                            [stemmer.stem(t.lower())
                             for t in word_tokenize(sentence)
                             if t.isalpha()])))

    return list((filter(lambda x: x.lower() not in english_stopwords and
                        x.lower() not in punctuation,
                        [t.lower() for t in word_tokenize(sentence)
                         if t.isalpha()])))

def clean(words):
    w = words.split(" ")
    a = list()
    english_stopwords = set(
        [stopword for stopword in stopwords.words('english')])
    punctuation = set(string.punctuation)
    punctuation.update(["``", "`", "...", ""])
    stemmer = SnowballStemmer('english')
    for word in w:
        #if (word.startswith("http://www.youtube.com") or word.startswith("https://www.youtube.com")):
        #    word = findTitle(word).lower()
        #    for arr in re.split("[, \-\()/’~'*0123456789!?:.;\"]+", word):
        #        if arr not in english_stopwords and arr not in punctuation and len(arr)>1:
        #           a.append(stemmer.stem(arr))
        if word.startswith("http"):
            a.append('{uri.netloc}'.format(uri=urlparse(word))) 
        else: 
            word = word.lower()
            #print(word)
            for arr in re.split("[, \-\()/’~'*0123456789!?:.;\"]+", word):
                if arr not in english_stopwords and arr not in punctuation and len(arr)>1:
                    a.append(stemmer.stem(arr))
    return ' '.join(a)


def token_processor(tokens):
    """ A custom token processor

    This function can be edited to add some additional
    transformation on the extracted tokens (e.g. stemming)

    At present, this function just passes the tokens through.
    """
    for t in tokens:
        yield t
        
def strip_accents_unicode(s):
    try:
        s = unicode(s, 'utf-8')
    except NameError:  # unicode is a default on python 3
        pass
    s = unicodedata.normalize('NFD', s)
    s = s.encode('ascii', 'ignore')
    s = s.decode("utf-8")
    return str(s)

class FeatureExtractor(TfidfVectorizer):

    def __init__(self):
        super(FeatureExtractor, self).__init__(
            input='content', encoding='utf-8',
            decode_error='strict', strip_accents=None, lowercase=True,
            preprocessor=None, tokenizer=None, analyzer='word',
            stop_words='english', token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 1), max_df=1.0, min_df=1,
            max_features=7000, vocabulary=None, binary=False,
            dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
            sublinear_tf=False)

    def fit(self, X_df, y=None):
        
        X_df.loc[:,'posts']= X_df.loc[:,'posts'].apply(lambda x: strip_accents_unicode(x)) #avoid unicode problem
        X_df.loc[:,'posts'] = X_df.loc[:,'posts'].apply(lambda x: x.split("|||"))
        X_df.loc[:,'posts']  = X_df.loc[:,'posts'].apply(lambda click: [(word.replace('"'," ")).replace("'","") 
                                                                  for word in click])
        X_df.loc[:,'posts']  = X_df.loc[:,'posts'].apply(lambda click: [clean(words) for words in click])
        X_df.loc[:,'posts']  = X_df.loc[:,'posts'].apply(lambda click: ' '.join(click))
        

        self._feat = np.array([' '.join(
            clean_str(strip_accents_unicode(dd)))
            for dd in X_df.posts]) #
        super(FeatureExtractor, self).fit(self._feat)
        
        return self

    def fit_transform(self, X_df, y=None):
        return self.fit(X_df).transform(X_df)

    def transform(self, X_df):
        X_df.loc[:,'posts']= X_df.loc[:,'posts'].apply(lambda x: strip_accents_unicode(x)) #avoid unicode problem
        
        X = super(FeatureExtractor, self).transform(X_df.posts) #
        X = X.todense()

        return X

    def build_tokenizer(self):
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: list(token_processor(tokenize(doc)))