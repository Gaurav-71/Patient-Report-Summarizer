import os
from nltk.corpus.reader.wordnet import path_similarity
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def preprocess_document(fpath):
    
    doc = open(fpath, encoding='iso-8859-1')
    num_words=0

    x = ""

    for line in doc:
        split_line = line.split(" ")
        num_words+=len(split_line)

        line = ''.join(e for e in line if e.isalnum() or e == ' ')

        line = line + "\n"
        x = x + line


    text_tokens = word_tokenize(x)
    tokens_without_stopwords = [
        word for word in text_tokens if word not in stopwords.words()]


    porter_stemmer = PorterStemmer()

    final_text = ""
    for t in tokens_without_stopwords:
        final_text = final_text + porter_stemmer.stem(t) + " "

    
    return final_text


def preprocess_query(query):
    line = query

    line = ''.join(e for e in line if e.isalnum() or e == ' ')

    text_tokens = word_tokenize(line)
    tokens_without_stopwords = [
        word for word in text_tokens if not word in stopwords.words()]

    x = ""
    for token in tokens_without_stopwords:
        x = x + token + " "

    porter_stemmer = PorterStemmer()

    text_tokens = word_tokenize(x)

    final_text = ""
    for w in text_tokens:
        final_text = final_text + porter_stemmer.stem(w) + " "

    return final_text