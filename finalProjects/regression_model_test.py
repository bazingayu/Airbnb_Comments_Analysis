import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA

df = pd.read_csv("../data/listings_cleaned_processed_amenities.csv")

df = df.dropna(subset=[ 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value'])

li = ['review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value']

arr = np.array(df)
dic = pickle.load(open("../data/reviews_processed_tf-idf.pkl", "rb"))
dic_listing = {}
for row in arr:
    dic_listing[row[0]] = row
X = []
Y = []
for item in dic:
    if item not in arr[:, 0]:
        print(item)
    else:
        X.append(dic[item])
        Y.append(dic_listing[item][14])
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
# from sklearn.linear_model import LinearRegression
# model = LinearRegression().fit(X, Y)
# print(model.intercept_, model.coef_)
pca = PCA(n_components=100)
X = pca.fit_transform(X)

for item in li:
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.1)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    ypred_train = model.predict(Xtrain)
    from sklearn.metrics import mean_squared_error
    print(ytest[:5], ypred[:5])

    print("train square error %f" % (mean_squared_error(ypred_train, ytrain)))
    print("test square error %f" % (mean_squared_error(ytest,ypred)))
