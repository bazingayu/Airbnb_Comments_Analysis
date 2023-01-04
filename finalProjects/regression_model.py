import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

df = pd.read_csv("../data/final_features_tf-idf.csv")
# df = pd.read_csv("../data/final_features.csv")


# X  = df.drop(columns=['listing_id', 'description', 'neighborhood_overview',
#        'review_scores_rating', 'review_scores_accuracy',
#        'review_scores_cleanliness', 'review_scores_checkin',
#        'review_scores_communication', 'review_scores_location',
#        'review_scores_value', 'host_response_time', 'host_response_rate', 'host_acceptance_rate',
#        'host_is_superhost', 'host_listings_count', 'host_total_listings_count',
#        'price', 'number_of_reviews', 'number_of_reviews_ltm',
#        'number_of_reviews_l30d', 'reviews_per_month',], axis=0)

# X  = df.drop(columns=['listing_id', 'description', 'neighborhood_overview',
#        'review_scores_rating', 'review_scores_accuracy',
#        'review_scores_cleanliness', 'review_scores_checkin',
#        'review_scores_communication', 'review_scores_location',
#        'review_scores_value'])
# X  = df.drop(columns=['listing_id', 'description', 'neighborhood_overview',
#        'review_scores_rating', 'review_scores_accuracy',
#        'review_scores_cleanliness', 'review_scores_checkin',
#        'review_scores_communication', 'review_scores_location',
#        'review_scores_value', 'host_response_time',
#        'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
#        'host_listings_count', 'host_total_listings_count', 'price',
#        'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d', 'reviews_per_month'])
# X = df[['host_response_time',
#        'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
#        'host_listings_count', 'host_total_listings_count', 'price',
#        'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d']]

# word2vec_feature_name = []
# for c in ['w2v_'+str(x) for x in range(300)]:
#        word2vec_feature_name.append(c)
# X = df[word2vec_feature_name]

tf_features_name = []
for item in df.keys():
       print(item)
       if item.find("tf-comments-") != -1 :
              tf_features_name.append(item)
X = df[tf_features_name]
# X = X.drop(columns=word2vec_feature_name)

# print(X.shape)
#
print(df['review_scores_rating'].isna().sum())

# print(X.keys())
li = [ 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value']
print(len(X.keys()))


for i in range(7):
       Y = np.array(df[li[i]])
       print(li[i])
       # print(li[i])
       # pca = PCA(n_components=100)
       # X = pca.fit_transform(X)
       from sklearn.model_selection import train_test_split
       Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.1)
       from sklearn.linear_model import LinearRegression, Lasso, Ridge
       # model = LinearRegression().fit(Xtrain, ytrain)
       # model = Lasso(alpha=0.5).fit(Xtrain, ytrain)
       model = Ridge(alpha=1 / (2 * 0.005))
       model.fit(Xtrain, ytrain)
       ypred = model.predict(Xtest)

       ypred_train = model.predict(Xtrain)

       from sklearn.metrics import mean_squared_error

       print(ytest[:5], ypred[:5])
       # print(X.keys())
       # print(model.coef_)
       print("square error %f" % (mean_squared_error(ypred_train, ytrain)))
       print("square error %f" % (mean_squared_error(ytest , ypred)))