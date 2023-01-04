import pandas as pd
import xlsxwriter.shape
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/final_features_word2vec.csv")
#
plt.figure(figsize=(50, 60))
X  = df.drop(columns=['listing_id',
       'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value'])
keys = X.keys()
# to_drop = []
# for item in keys:
#        if "tf-" in item or 'w2v' in item or 'amen' in item:
#             to_drop.append(item)
# X = X.drop(columns=to_drop)
keys = X.keys()
X = np.array(X)
#
print(df['review_scores_rating'].isna().sum())

li = [ 'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value']
for index in range(len(li)):
       y = np.array(df[li[index]])
       print(X.shape, y.shape)

       mean_error=[]; std_error=[]

       from sklearn.linear_model import Ridge
       model = Ridge(alpha=1/(2*0.005))
       temp=[]
       from sklearn.model_selection import train_test_split

       Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.1)

       model.fit(Xtrain, ytrain)
       c = np.array(model.coef_)
       sort_keys = np.absolute(np.array(model.coef_)).argsort()
       value = c[sort_keys[-20:]]
       x = keys[sort_keys[-20:]]

       plt.subplot(4, 2, index+1)
       plt.title(li[index], fontsize=50)
       color = ['b' if i > 0 else 'r' for i in value]
       value = np.absolute(value)
       plt.barh(x, value,color=color)
       plt.xticks(fontsize=20)
       plt.yticks(fontsize=50)
plt.tight_layout()
plt.savefig("test.png")
# plt.show()


