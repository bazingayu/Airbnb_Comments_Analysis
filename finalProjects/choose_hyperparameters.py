# import pandas as pd
# import xlsxwriter.shape
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import numpy as np
#
# df = pd.read_csv("../data/final_features_word2vec.csv")
# #
#
# X  = df.drop(columns=['listing_id',
#        'review_scores_rating', 'review_scores_accuracy',
#        'review_scores_cleanliness', 'review_scores_checkin',
#        'review_scores_communication', 'review_scores_location',
#        'review_scores_value'])
#
# X = np.array(X)
# # pca = PCA(n_components=50)
# # X = pca.fit_transform(X)
# #
# print(df['review_scores_rating'].isna().sum())
#
# li = [ 'review_scores_rating', 'review_scores_accuracy',
#        'review_scores_cleanliness', 'review_scores_checkin',
#        'review_scores_communication', 'review_scores_location',
#        'review_scores_value']
# y = np.array(df['review_scores_rating'])
# print(X.shape, y.shape)
#
# mean_error=[]; std_error=[]
# Ci_range = [0.001, 0.005, 0.01, 0.05, 0.1]
# for Ci in Ci_range:
#        from sklearn.linear_model import Ridge
#        model = Ridge(alpha=1/(2*Ci))
#        temp=[]
#        from sklearn.model_selection import KFold
#        kf = KFold(n_splits=5)
#        for train, test in kf.split(X):
#               model.fit(X[train], y[train])
#               ypred = model.predict(X[test])
#               from sklearn.metrics import mean_squared_error
#               temp.append(mean_squared_error(y[test],ypred))
#        mean_error.append(np.array(temp).mean())
#        std_error.append(np.array(temp).std())
# import matplotlib.pyplot as plt
# plt.errorbar(Ci_range,mean_error,yerr=std_error)
# plt.xlabel('Ci'); plt.ylabel('Mean square error')
# plt.xlim((0,0.1))
# plt.show()



import pandas as pd
import xlsxwriter.shape
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

df = pd.read_csv("../data/final_features_word2vec.csv")
#

X  = df.drop(columns=['listing_id',
       'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value'])

X = np.array(X)
# pca = PCA(n_components=50)
# X = pca.fit_transform(X)
#
print(df['review_scores_rating'].isna().sum())

li = [ 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value']
y = np.array(df['review_scores_rating'])
print(X.shape, y.shape)

mean_error=[]; std_error=[]
Ci_range = [5, 50, 100, 150, 200]
for Ci in Ci_range:
       from sklearn.neighbors import KNeighborsRegressor
       model = KNeighborsRegressor(n_neighbors=Ci)
       temp=[]
       from sklearn.model_selection import KFold
       kf = KFold(n_splits=5)
       for train, test in kf.split(X):
              model.fit(X[train], y[train])
              ypred = model.predict(X[test])
              from sklearn.metrics import mean_squared_error
              temp.append(mean_squared_error(y[test],ypred))
       mean_error.append(np.array(temp).mean())
       std_error.append(np.array(temp).std())
import matplotlib.pyplot as plt
plt.errorbar(Ci_range,mean_error,yerr=std_error)
plt.xlabel('K'); plt.ylabel('Mean square error')
plt.xlim((0,200))
plt.show()