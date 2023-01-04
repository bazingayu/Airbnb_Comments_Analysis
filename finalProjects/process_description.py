import pandas as pd
import re
from tqdm import tqdm
import spacy
import nltk
import numpy as np

nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# df = pd.read_csv("../data/final_features_word2vec_tf-idf.csv")
df['description'] = df['description'].fillna("")
df['description'] = df['description'].apply(lambda x : re.sub(r"\(.*?\)|\<.*?\>|\[.*?\]", "", string=x))

# remove puctuation marks
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
for punc in punctuation:
    df['description'] = df['description'].apply(lambda x : x.replace(punc, " "))
# reviews['comments'] = reviews['comments'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
def get_solo(text):
    duels=[x+x for x in list(',.')]
    duels.append(',.')
    duels.append('.,')
    #如需增加标点符号,比如问号,直接将list('。，!')换成list('。，!？')即可.
    for d in duels:
        while d in text:
            text=text.replace(d,d[0])
    return text
df['description'] = df['description'].apply(lambda x : get_solo(x))
print(df.head())
print("removed puctuation marks")
# convert text to lowercase
df['description'] = df['description'].str.lower()
print(df['description'].head())
print('converted to lowercase')
#remove numbers
df['description'] = df['description'].str.replace("[0-9]", " ")
print(df['description'].head())
print("removed numbers")
# remove whitespaces
df['description'] = df['description'].apply(lambda x:' '.join(x.split()))
print(df['description'].head())
print("removed whitespaces")
df['description'] = df['description'].apply(lambda x: ' '.join(word_tokenize(x)))
print(df['description'].head())
print("word tokenize")

def stemer(texts):
    output = []
    for i in tqdm(texts):
        if i is None:
            output.append(".")
        s = [stemmer.stem(token) for token in i]
        output.append(''.join(s))
    return output
# use spacy to lemmatize text
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sw_spacy = nlp.Defaults.stop_words
def lemmatization(texts):
    output = []
    for i in tqdm(texts):
        # print(i)
        s = [token.lemma_ for token in nlp(i) if token.is_stop is False]
        output.append(' '.join(s))
    return output
df['description'] = stemer(df['description'])
# # lemmatized
df['description'] = lemmatization(df['description'])
print(df['description'].head())
print("finish lemmatized")
print("finish preprocessing")


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=0.1, max_df=0.8, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['description'])
print(len(vectorizer.get_feature_names()))

print(vectorizer.get_feature_names())
X = X.toarray()

trainDataVecs = np.array(X)
print(trainDataVecs.shape)
# for items in amenities_list:
#     df[items] = df["amenities"].apply(lambda x: 1 if items in x else 0)
for i, x in enumerate(vectorizer.get_feature_names()):
    df["tf-des-" + x] = trainDataVecs[:, i]
    # train_data_vecs_df = pd.DataFrame(data=trainDataVecs, columns=["tf-des-" + x for x in vectorizer.get_feature_names()])
print(df.keys())

# print(train_data_vecs_df.head())
#
# train_data_vecs_df.to_csv("../data/reviews_tf-idf.csv", index=False)

df['neighborhood_overview'] = df['neighborhood_overview'].fillna("")
df['neighborhood_overview'] = df['neighborhood_overview'].apply(lambda x : re.sub(r"\(.*?\)|\<.*?\>|\[.*?\]", "", string=x))

# remove puctuation marks
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
for punc in punctuation:
    df['neighborhood_overview'] = df['neighborhood_overview'].apply(lambda x : x.replace(punc, " "))
# reviews['comments'] = reviews['comments'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
def get_solo(text):
    duels=[x+x for x in list(',.')]
    duels.append(',.')
    duels.append('.,')
    #如需增加标点符号,比如问号,直接将list('。，!')换成list('。，!？')即可.
    for d in duels:
        while d in text:
            text=text.replace(d,d[0])
    return text
df['neighborhood_overview'] = df['neighborhood_overview'].apply(lambda x : get_solo(x))
print(df.head())
print("removed puctuation marks")
# convert text to lowercase
df['neighborhood_overview'] = df['neighborhood_overview'].str.lower()
print(df['neighborhood_overview'].head())
print('converted to lowercase')
#remove numbers
df['neighborhood_overview'] = df['neighborhood_overview'].str.replace("[0-9]", " ")
print(df['neighborhood_overview'].head())
print("removed numbers")
# remove whitespaces
df['neighborhood_overview'] = df['neighborhood_overview'].apply(lambda x:' '.join(x.split()))
print(df['neighborhood_overview'].head())
print("removed whitespaces")
df['neighborhood_overview'] = df['neighborhood_overview'].apply(lambda x: ' '.join(word_tokenize(x)))
print(df['neighborhood_overview'].head())
print("word tokenize")

def stemer(texts):
    output = []
    for i in tqdm(texts):
        if i is None:
            output.append(".")
        s = [stemmer.stem(token) for token in i]
        output.append(''.join(s))
    return output
# use spacy to lemmatize text
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sw_spacy = nlp.Defaults.stop_words
def lemmatization(texts):
    output = []
    for i in tqdm(texts):
        # print(i)
        s = [token.lemma_ for token in nlp(i) if token.is_stop is False]
        output.append(' '.join(s))
    return output
df['neighborhood_overview'] = stemer(df['neighborhood_overview'])
# # lemmatized
df['neighborhood_overview'] = lemmatization(df['neighborhood_overview'])
print(df['neighborhood_overview'].head())
print("finish lemmatized")
print("finish preprocessing")


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=0.1, max_df=0.8, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df['neighborhood_overview'])
print(len(vectorizer.get_feature_names()))

print(vectorizer.get_feature_names())
X = X.toarray()

trainDataVecs = np.array(X)
print(trainDataVecs.shape)
# for items in amenities_list:
#     df[items] = df["amenities"].apply(lambda x: 1 if items in x else 0)
for i, x in enumerate(vectorizer.get_feature_names()):
    df["tf-nei-" + x] = trainDataVecs[:, i]
    # train_data_vecs_df = pd.DataFrame(data=trainDataVecs, columns=["tf-des-" + x for x in vectorizer.get_feature_names()])
print(df.keys())
df.drop(columns="description",inplace=True)
df.drop(columns="neighborhood_overview",inplace=True)
# print(train_data_vecs_df.head())
#
# train_data_vecs_df.to_csv("../data/reviews_tf-idf.csv", index=False)




df.to_csv("../data/final_features_description_neighboardhood.csv")


