from collections import Counter
import numpy as np
import pickle
import dates
import preprocessing
from nltk.tokenize import word_tokenize


index = None
vocabulary = None
document_lengths = None
with open('vocabulary', 'rb') as fp:
    vocabulary = pickle.load(fp)
with open('index', 'rb') as fp:
    index = pickle.load(fp)
with open('document_lengths', 'rb') as fp:
    document_lengths = pickle.load(fp)


tf_matrix = np.zeros((len(dates.files), len(vocabulary)))


def find_tf():
    for key in index:
        term = index[key]
        for tf in term[1].keys():
            term_freq = term[1][tf]
            i = vocabulary.index(key)
            tf_matrix[tf][i] = term_freq/document_lengths[tf]


find_tf()
idf_matrix = np.zeros(len(vocabulary))


def find_idf():
    for key in index:
        term = index[key]
        idf = len(term[1].keys())
        res = np.log10(10/idf)
        i = vocabulary.index(key)
        idf_matrix[i] = res


find_idf()

tf_idf = tf_matrix * idf_matrix

def query_tf_idf(query):
    counter = Counter(query)
    word_count = len(query)
    tfm = np.zeros(len(vocabulary))
    idfm = np.zeros(len(vocabulary))
    for token in np.unique(query):
        tf = counter[token]/word_count
        try:
            i = vocabulary.index(token)
        except:
            continue
        idfm = idf_matrix[i]
        tfm[i] = tf
    return(tfm*idfm)


def print_ranks(ranking):
    print("\n\nDocument Number \t Rank")
    print("--------------------------------------------------")
    for pair in ranking:
        print("\t",pair[1],"\t\t",pair[0])
    print("--------------------------------------------------\n\n")


def vsm(listDocs, query):

    query = preprocessing.preprocess_query(query)
    query = word_tokenize(query)
    query_matrix = query_tf_idf(query)

    ranking = []
    for i in range(0, len(listDocs)):
        row = tf_idf[i]
        rank = np.dot(row, np.transpose(query_matrix)) / \
            (np.linalg.norm(row)*np.linalg.norm(query_matrix))
        ranking.append([rank, i])

    ranking.sort(reverse=True)
    print_ranks(ranking)

    return ranking


