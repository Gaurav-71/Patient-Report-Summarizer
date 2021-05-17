# Expanding Contractions for frequently used shortforms
import html
from sys import path
import numpy as np
import re
import networkx
from nltk.corpus import wordnet as wn
from pattern3.en import tag
import unicodedata
import string
import nltk
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}


def build_feature_matrix(documents):
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1))
    feature_matrix = vectorizer.fit_transform(documents).astype(float)
    print("--------------------------FFFF", feature_matrix)
    return vectorizer, feature_matrix


stopword_list = nltk.corpus.stopwords.words('english')


# tokenize the report into tokens
def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens


# Match the shortforms used in the report by doctors and replace them with the correct words
def expand_contractions(text, contraction_mapping):
    contractions_pattern = re.compile('({})'.format(
        '|'.join(contraction_mapping.keys())), flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
            if contraction_mapping.get(match)\
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# to eliminate special caracters from report
def remove_special_characters(text):
    tokens = tokenize_text(text)
    # string.punctuation = !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(
        None, [pattern.sub(' ', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# to eliminate stop words which do not provide any useful info
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# to remove any html related syntax
def unescape_html(parser, text):
    return parser.unescape(text)


# normalization of text
def normalize_corpus(corpus):
    normalized_corpus = []
    for text in corpus:
        text = html.unescape(text)
        text = expand_contractions(text, CONTRACTION_MAP)
        text = text.lower()
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized_corpus.append(text)
    return normalized_corpus


# parse the document to check non ascii characters
def parse_document(document):
    document = re.sub('\n', ' ', document)
    if isinstance(document, str):
        document = document
    elif isinstance(document, unicode):
        return unicodedata.normalize('NFKD', document).encode('ascii', 'ignore')
    else:
        raise ValueError('Document is not string or unicode!')
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences


def textrank_text_summarizer(documents, sentences, num_sentences=2):
    vec, dt_matrix = build_feature_matrix(documents)
    similarity_matrix = (dt_matrix * dt_matrix.T)
    similarity_graph = networkx.from_scipy_sparse_matrix(similarity_matrix)
    scores = networkx.pagerank(similarity_graph)
    ranked_sentences = sorted(((score, index)
                               for index, score in scores.items()), reverse=True)
    top_sentence_indices = [ranked_sentences[index][1]
                            for index in range(num_sentences)]
    top_sentence_indices.sort()
    finalResult = []
    for index in top_sentence_indices:
        finalResult.append(sentences[index])
    return finalResult


def summarisefile(filename):
    path = "./static/Data/"+filename + ".txt"
    i = open(path)
    my_text = i.read()
    summary_sentences = parse_document(my_text)
    normalized_sentences = normalize_corpus(summary_sentences)
    print("Total Sentences:", len(normalized_sentences))
    print("---------text-rank summarization for document--------")
    summaryArr = textrank_text_summarizer(
        normalized_sentences, summary_sentences, num_sentences=5)
    summary = " ".join(summaryArr)
    print(summaryArr)
    return summaryArr


def filecontent(filename):
    path = "./static/Data/" + filename + ".txt"
    content = open(path).readlines()
    return content
