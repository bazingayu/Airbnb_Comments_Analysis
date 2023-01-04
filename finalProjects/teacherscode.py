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

pd.set_option('display.max_colwidth', 200)

reviews = pd.read_csv("../data/reviews.csv", dtype={'comments': str})
print(reviews.keys())
print(reviews)
# DELETE NAN
reviews = reviews.dropna(axis=0, how='any')
def removeweblabel(texts):
    output = []
    for i in tqdm(texts):
        s = i.replace("\r", ",")
        s = s.replace("<br/>", ",")
        output.append(s)

    return output
reviews['comments'] = removeweblabel(reviews['comments'])
print(reviews['comments'])
# remove puctuation marks
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
reviews['comments'] = reviews['comments'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

print("start")
print(reviews.head())
print("end")
print("removed puctuation marks")
reviews.to_csv("./reviews_preprocessing_removed_puctuation_marks.csv", index=False)



def translatefunc(texts):
    output = []
    translator = Translator(service_urls=['translate.google.com', ])  # 如果可以上外网，还可添加 'translate.google.com' 等
    translator.raise_Exception = True
    all_false = 0
    all_ = 0
    for i in tqdm(texts):
        if i == '':
            output.append(i)
            continue
        try:
            lang = detect(i)
            if lang != 'en':
                all_ += 1
                try:
                    output.append(translator.translate(i, dest='en'))
                except:
                    time.sleep(10)
                    output.append(i)
                    all_false += 1
                    print(f"Error occured {all_false}, {all_}")
            else:
                output.append(i)
        except:
            output.append(str(i))
    with open("translated.txt", "w") as f:
        for i in output:
            try:
                f.write(i + "\n")
            except:
                print("save false")
    return output


reviews['comments'] = translatefunc(reviews['comments'])
reviews.to_csv("./reviews_preprocessing_translated.csv", index=False)
# convert text to lowercase
reviews['comments'] = reviews['comments'].str.lower()
print('converted to lowercase')
#remove numbers
reviews['comments'] = reviews['comments'].str.replace("[0-9]", " ")
print("removed numbers")
# remove whitespaces
reviews['comments'] = reviews['comments'].apply(lambda x:' '.join(x.split()))
# stemming
def stemer(texts):
    output = []
    for i in tqdm(texts):
        s = [stemmer.stem(token) for token in i]
        output.append(' '.join(s))
    return output
# use spacy to lemmatize text
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sw_spacy = nlp.Defaults.stop_words
def lemmatization(texts):
    output = []
    for i in tqdm(texts):
        s = [token.lemma_ for token in nlp(i) if token.is_stop is False]
        output.append(' '.join(s))
    return output
reviews['comments'] = stemer(reviews['comments'])
reviews['comments'] = lemmatization(reviews['comments'])

# lemmatized
reviews.to_csv("./reviews_preprocessing.csv", index=False)
print(reviews['comments'].head())

