import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import word2vec, Word2Vec
import pickle

num_features = 300
min_word_count = 50
num_workers = 4
context = 5
downsampling = 0.001
reviews = pd.read_csv("../data/reviews_preprocessing_test.csv", dtype={'comments': str})
# reviews = reviews.groupby(['listing_id'])['comments'].apply(lambda x: ','.join(x)).reset_index()
reviews = reviews.dropna()
review_comments = reviews['comments']
sentences = []
# print(review_comments)
for review_comment in tqdm(review_comments):
    review_ = review_comment.replace(",", ".")
    review_ = review_.replace(".", " ")
    review_ = ' '.join(review_.split())
    reviews_ = review_.split(" ")
    # print(len(reviews_))
    # print(reviews_)
    # for r in reviews_:
    reviews_ = [value for value in reviews_ if value != ""]
    reviews_ = [value for value in reviews_ if value != " "]
    reviews_ = [value for value in reviews_ if value != "'"]
    sentences.append(reviews_)
    # sentences.append(review_)
# print(sentences)
print(len(sentences))


# print(sentences)
# model = Word2Vec.load('../data/model_fulllr_500_50_4_5_001.bin')
model = word2vec.Word2Vec(sentences, workers=num_workers, vector_size=num_features, min_count=min_word_count, window=context, sample=downsampling)
# model.save('../data/model_fulllr_500_50_4_5_001.bin')
# print(model.wv['excellent'])
for val in model.wv.similar_by_word("excellent", topn=10):
    print(val[0], val[1])
dic = {}
vocab = model.wv
print(vocab.index_to_key)
print(len(vocab.index_to_key))
trainDataVecs = []
reviews = pd.read_csv("../data/reviews_preprocessing.csv", dtype={'comments': str})
review_comments = reviews['comments']
sentences = []
# print(review_comments)
for review_comment in tqdm(review_comments):
    review_ = review_comment.replace(",", ".")
    review_ = review_.replace(".", " ")
    review_ = review_.replace("|", " ")
    review_ = ' '.join(review_.split())
    reviews_ = review_.split(" ")
    reviews_ = [value for value in reviews_ if value != ""]
    reviews_ = [value for value in reviews_ if value != " "]
    reviews_ = [value for value in reviews_ if value != "'"]
    sentences.append(reviews_)
    # sentences.append(review_)
# print(sentences)

listing_ids = reviews['listing_id']
for i in tqdm(range(reviews.shape[0])):
    vec = np.zeros(num_features)
    num = 0
    for word in sentences[i]:
        if word in vocab:
            vec += model.wv[word]
            num += 1
    vec /= num
    trainDataVecs.append(vec)
    dic[reviews['listing_id'].iloc[i]] = vec
trainDataVecs = np.array(trainDataVecs)
train_data_vecs_df = pd.DataFrame(data=trainDataVecs, columns=['w2v_'+str(x) for x in range(num_features)])
train_data_vecs_df['listing_id'] = listing_ids

train_data_vecs_df.to_csv("../data/reviews_word2vec.csv", index=False)


