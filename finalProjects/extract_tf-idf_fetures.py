import json
import pickle

import pandas as pd
import numpy as np
import xarray.core.utils
from tqdm import tqdm
# from correct_spelling import correct_text_generic

df = pd.read_csv("../data/reviews_preprocessing.csv", dtype={'comments': str})
df = df.fillna("")
listing_ids = df["listing_id"]
dic = {}

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.8, stop_words='english')
df = df.groupby(['listing_id'])['comments'].apply(lambda x: '|'.join(x)).reset_index()
X = vectorizer.fit_transform(df['comments'])
print(len(vectorizer.get_feature_names()))

print(vectorizer.get_feature_names())

X = X.toarray()

trainDataVecs = np.array(X)
print(trainDataVecs.shape)
train_data_vecs_df = pd.DataFrame(data=trainDataVecs, columns=["tf-comments-" + x for x in vectorizer.get_feature_names()])
train_data_vecs_df['listing_id'] = listing_ids
print(train_data_vecs_df.head())

train_data_vecs_df.to_csv("../data/reviews_tf-idf.csv", index=False)

