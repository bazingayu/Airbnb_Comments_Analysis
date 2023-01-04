import pandas as pd
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import time
import pickle
import spacy
import nltk
from googletrans import Translator
from langdetect import detect, LangDetectException

nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

pd.set_option('display.max_colwidth', 300)

reviews = pd.read_csv("../data/reviews_translated.csv", dtype={'comments': str})
# DELETE NAN
reviews = reviews.dropna(axis=0, how='any')

# remove puctuation marks
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
for punc in punctuation:
    reviews['comments'] = reviews['comments'].apply(lambda x : x.replace(punc, " "))
# reviews['comments'] = reviews['comments'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
def get_solo(text):
    # remove the continuous ,.
    duels=[x+x for x in list(',.')]
    duels.append(',.')
    duels.append('.,')
    for d in duels:
        while d in text:
            text=text.replace(d,d[0])
    return text
reviews['comments'] = reviews['comments'].apply(lambda x : get_solo(x))
print(reviews.head())
print("removed puctuation marks")


# convert text to lowercase
reviews['comments'] = reviews['comments'].str.lower()
print(reviews['comments'].head())
print('converted to lowercase')
#remove numbers
reviews['comments'] = reviews['comments'].str.replace("[0-9]", " ")
print(reviews['comments'].head())
print("removed numbers")
# remove whitespaces
reviews['comments'] = reviews['comments'].apply(lambda x:' '.join(x.split()))
print(reviews['comments'].head())
print("removed whitespaces")
reviews['comments'] = reviews['comments'].apply(lambda x: ' '.join(word_tokenize(x)))
print(reviews['comments'].head())
print("word tokenize")
# stemming
# def stemer(texts):
#     output = []
#     for i in tqdm(texts):
#         if i is None:
#             output.append(".")
#         s = [stemmer.stem(token) for token in i]
#         output.append(''.join(s))
#     return output
# use spacy to lemmatize text
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sw_spacy = nlp.Defaults.stop_words
def lemmatization(texts):
    output = []
    for i in tqdm(texts):
        s = [token.lemma_ for token in nlp(i) if token.is_stop is False]
        output.append(' '.join(s))
    return output
# reviews['comments'] = stemer(reviews['comments'])
# # lemmatized
reviews['comments'] = lemmatization(reviews['comments'])
print(reviews['comments'].head())
print("finish lemmatized")

# # delete whitespace
reviews['comments'] = reviews['comments'].apply(lambda x: ' '.join(x.split()))
reviews = reviews.dropna(subset=["comments"])
to_drop = ["id", "date", "reviewer_id", "reviewer_name"]
reviews = reviews.drop(columns=to_drop, axis=1)
# reviews = reviews.groupby(['listing_id'])['comments'].apply(lambda x: '|'.join(x)).reset_index()
reviews.to_csv("../data/reviews_preprocessing_test.csv", index=False)
print(reviews['comments'].head())

