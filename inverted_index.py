from collections import defaultdict
import pickle
import preprocessing
import numpy as np
import pandas as pd
import dates
from nltk.tokenize import word_tokenize

preprocessing_done = False

def preprocess_and_save(fileList):
    text = []
    document_lengths = []
    print('\nPreprocessing Documents...')
    for f in fileList:
        tx = preprocessing.preprocess_document("./static/Data/"+f)
        text_tokens = word_tokenize(tx)
        document_lengths.append(len(text_tokens))
        text.append(text_tokens)

    with open('document_lengths', 'wb') as fp:
        pickle.dump(document_lengths, fp)

    print('Done\n')

    return text


def create_index(data):
    index = defaultdict(list)
    for i, tokens in enumerate(data):
        for token in tokens:
            if(index[token]):
                index[token][0][0] = index[token][0][0] + 1
                if i in index[token][1].keys():
                    index[token][1][i] = index[token][1][i] + 1
                else:
                    index[token][1][i] = 1
            else:
                index[token] = [[1], {i: 1}]
    return index

text = []
if (not preprocessing_done):
    text = preprocess_and_save(dates.files)
    with open('text', 'wb') as fp:
        pickle.dump(text, fp)

with open('text', 'rb') as fp:
    text = pickle.load(fp)

with open('document_lengths', 'rb') as fp:
    document_lengths = pickle.load(fp)

index = create_index(text)

with open('vocabulary', 'wb') as fp:
    pickle.dump(list(index.keys()), fp)

with open('index', 'wb') as fp:
    pickle.dump(index, fp)



