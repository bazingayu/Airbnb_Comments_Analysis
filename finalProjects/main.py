import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import time
import pickle
import spacy
from googletrans import Translator
translator = Translator(service_urls=['translate.google.com',])# 如果可以上外网，还可添加 'translate.google.com' 等


pd.set_option('display.max_colwidth', 200)

reviews = pd.read_csv("../data/reviews.csv", dtype={'comments': str})
print(reviews.keys())
print(reviews)
# DELETE NAN
reviews = reviews.dropna(axis=0, how='any')
# remove puctuation marks
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
reviews['comments'] = reviews['comments'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
print("removed puctuation marks")

# convert text to lowercase
reviews['comments'] = reviews['comments'].str.lower()
print('converted to lowercase')
#remove numbers
reviews['comments'] = reviews['comments'].str.replace("[0-9]", " ")
print("removed numbers")

# remove whitespaces
reviews['comments'] = reviews['comments'].apply(lambda x:' '.join(x.split()))
# removed whitespaces
# use spacy to lemmatize text
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sw_spacy = nlp.Defaults.stop_words
def lemmatization(texts):
    output = []
    for i in tqdm(texts):
        s = [token.lemma_ for token in nlp(i) if token.is_stop is False]
        output.append(' '.join(s))
    return output
reviews['comments'] = lemmatization(reviews['comments'])
# lemmatized
reviews.to_csv("./reviews_preprocessing1.csv", index=False)
print(reviews['comments'].head())