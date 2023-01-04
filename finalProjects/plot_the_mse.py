import pandas as pd
import xlsxwriter.shape
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize

df = pd.read_csv("../data/final_features_word2vec.csv")
print(df.shape)
#

X  = df.drop(columns=['listing_id',
       'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value'])

X = np.array(X)

print(df['review_scores_rating'].isna().sum())

li = [ 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value']
names = [x.split('_')[-1] for x in li]

KNN_mean_error = []
KNN_std_error = []
ridge_mean_error = []
ridge_std_error = []
dummy_mean_error = []
dummy_std_error = []
for index, item in tqdm(enumerate(li)):
       from sklearn.neighbors import KNeighborsRegressor
       y = df[li[index]]
       model = KNeighborsRegressor(n_neighbors=100)
       temp_knn=[]
       temp_ridge = []
       temp_dummy = []
       from sklearn.model_selection import KFold
       kf = KFold(n_splits=5)
       for train, test in kf.split(X):
              model.fit(X[train], y[train])
              ypred = model.predict(X[test])
              from sklearn.metrics import mean_squared_error
              temp_knn.append(mean_squared_error(y[test],ypred))
       # temp_knn = normalize(np.array(temp_knn)[:, np.newaxis], axis=0).ravel()
       KNN_mean_error.append(np.array(temp_knn).mean())
       KNN_std_error.append(np.array(temp_knn).std())


       from sklearn.linear_model import Ridge
       model = Ridge(alpha=1/(2*0.005))
       from sklearn.model_selection import KFold
       kf = KFold(n_splits=5)
       for train, test in kf.split(X):
              model.fit(X[train], y[train])
              ypred = model.predict(X[test])
              from sklearn.metrics import mean_squared_error
              temp_ridge.append(mean_squared_error(y[test],ypred))
       # temp_ridge = normalize(np.array(temp_ridge)[:, np.newaxis], axis=0).ravel()
       ridge_mean_error.append(np.array(temp_ridge).mean())
       ridge_std_error.append(np.array(temp_ridge).std())

       from sklearn.dummy import DummyRegressor
       model = DummyRegressor(strategy="mean", constant=None)
       from sklearn.model_selection import KFold

       kf = KFold(n_splits=5)
       for train, test in kf.split(X):
           model.fit(X[train], y[train])
           ypred = model.predict(X[test])
           from sklearn.metrics import mean_squared_error

           temp_dummy.append(mean_squared_error(y[test], ypred))
       # temp_dummy = normalize(np.array(temp_dummy)[:, np.newaxis], axis=0).ravel()
       dummy_mean_error.append(np.array(temp_dummy).mean())
       dummy_std_error.append(np.array(temp_dummy).std())


df = pd.read_csv("../data/final_features_tf-idf.csv")
print(df.shape)

X  = df.drop(columns=['listing_id',
       'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value'])

X = np.array(X)

print(df['review_scores_rating'].isna().sum())

li = [ 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value']
names = [x.split('_')[-1] for x in li]

KNN_tf_mean_error = []
KNN_tf_std_error = []
ridge_tf_mean_error = []
ridge_tf_std_error = []
for index, item in tqdm(enumerate(li)):
       from sklearn.neighbors import KNeighborsRegressor
       y = df[li[index]]
       model = KNeighborsRegressor(n_neighbors=100)
       temp_knn=[]
       temp_ridge = []
       from sklearn.model_selection import KFold
       kf = KFold(n_splits=5)
       for train, test in kf.split(X):
              model.fit(X[train], y[train])
              ypred = model.predict(X[test])
              from sklearn.metrics import mean_squared_error
              temp_knn.append(mean_squared_error(y[test],ypred))
       # temp_knn = normalize(np.array(temp_knn)[:, np.newaxis], axis=0).ravel()
       KNN_tf_mean_error.append(np.array(temp_knn).mean())
       KNN_tf_std_error.append(np.array(temp_knn).std())


       from sklearn.linear_model import Ridge
       model = Ridge(alpha=1/(2*0.005))
       from sklearn.model_selection import KFold
       kf = KFold(n_splits=5)
       for train, test in kf.split(X):
              model.fit(X[train], y[train])
              ypred = model.predict(X[test])
              from sklearn.metrics import mean_squared_error
              temp_ridge.append(mean_squared_error(y[test],ypred))
       # temp_ridge = normalize(np.array(temp_ridge)[:, np.newaxis], axis=0).ravel()
       ridge_tf_mean_error.append(np.array(temp_ridge).mean())
       ridge_tf_std_error.append(np.array(temp_ridge).std())




import matplotlib.pyplot as plt
# plt.errorbar(names,KNN_mean_error,yerr=KNN_std_error)
# plt.errorbar(names,ridge_mean_error,yerr=ridge_std_error)
# plt.errorbar(names,KNN_tf_mean_error,yerr=KNN_tf_std_error)
# plt.errorbar(names,ridge_tf_mean_error,yerr=ridge_tf_std_error)
# plt.errorbar(names,dummy_mean_error,yerr=dummy_std_error)
plt.plot(names, KNN_mean_error)
plt.plot(names, ridge_mean_error)
plt.plot(names, KNN_tf_mean_error)
plt.plot(names, ridge_tf_mean_error)
plt.plot(names, dummy_mean_error)
plt.legend(["knn-word2vec", "ridge-word2vec", "knn-tf-idf", "ridge-tf-idf", "dummy"], loc ="upper right", fontsize=20)
plt.xticks(size=20)
plt.yticks(size=20)
# plt.ylim(0.42, 0.46)
plt.xlabel('reviews-items', fontsize=20); plt.ylabel('Mean square error', fontsize=20)
plt.show()