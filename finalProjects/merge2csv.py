import os
import pandas as pd

listings = pd.read_csv("../data/listings_features.csv")
reviews = pd.read_csv("../data/reviews_tf-idf.csv")

listings = listings.rename(columns={'id':'listing_id'})
listings = listings.merge(reviews, on='listing_id', how='left')

# word2vec_feature_name = []
# for c in ['w2v_'+str(x) for x in range(300)]:
#        word2vec_feature_name.append(c)
#
listings = listings.dropna()
# listings = listings.dropna(subset=[ 'review_scores_rating', 'review_scores_accuracy',
#        'review_scores_cleanliness', 'review_scores_checkin',
#        'review_scores_communication', 'review_scores_location',
#        'review_scores_value'])
print(listings.shape)
listings.to_csv("../data/final_features_tf-idf.csv", index=False)



# listings = pd.read_csv("../data/listings_cleaned_processed_amenities.csv")
# reviews = pd.read_csv("../data/reviews_tf-idf.csv")
#
# listings = listings.rename(columns={'id':'listing_id'})
# listings = listings.merge(reviews, on='listing_id', how='left')
#
# # word2vec_feature_name = []
# # for c in ['w2v_'+str(x) for x in range(300)]:
# #        word2vec_feature_name.append(c)
# #
# listings = listings.dropna()
# listings = listings.dropna(subset=[ 'review_scores_rating', 'review_scores_accuracy',
#        'review_scores_cleanliness', 'review_scores_checkin',
#        'review_scores_communication', 'review_scores_location',
#        'review_scores_value'])
# listings.to_csv("../data/final_features_tf-idf.csv", index=False)