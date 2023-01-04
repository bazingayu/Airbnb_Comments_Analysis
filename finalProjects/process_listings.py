import pandas as pd
import numpy as np
from itertools import chain
import warnings
warnings.filterwarnings("ignore")
import re
from tqdm import tqdm
import spacy
import nltk
import numpy as np

nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
# process host_response_time
listings = pd.read_csv("../data/listings.csv")
keys = listings.keys()

# host neighboardhood, host_has_profile_pic, host_identity_verified,    property_type, room_type, accommodates, bathrooms_text,   bedrooms, beds, availability_30,
#availability_60, availability_90, availability_365, calculated_host_listings_count,   calculated_host_listings_count_entire_homes, calculated_host_listings_count_private_rooms
#calculated_host_listings_count_shared_rooms,
# delete unrelated columes
to_drop = ["listing_url", "scrape_id", "last_scraped", "source", "name", "picture_url", "host_id", "host_location", "host_url", "host_name", "host_since", "host_about",\
           "host_thumbnail_url", "host_picture_url", "host_verifications",\
           "neighbourhood", "neighbourhood_cleansed", "neighbourhood_group_cleansed", "latitude", "longitude", \
           "bathrooms", "minimum_nights", "maximum_nights", "minimum_minimum_nights", \
           "maximum_minimum_nights", "minimum_maximum_nights", "maximum_maximum_nights", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm", "calendar_updated",
           "has_availability",  "calendar_last_scraped", "first_review",\
           "last_review", "license", "instant_bookable"]
print(to_drop)
listings = listings.drop(columns=to_drop, axis=1)
print(listings.keys())
print(listings.head())
listings.to_csv("../data/listings_cleaned.csv", index=False)

# process some values

def process_amenities(item):
    if item == None:
        return None
    if item == "":
        return ""
    item = item.lower()
    item = item.replace("u2013", "")
    item = item.replace("u2019n", "")
    item = item.replace("2019s", "")
    item = item.replace("Lu2019Oru00e9al ", "")
    item = item.replace("L'Oru00e9al ", "")
    item = item.replace("u00a0In ", "")
    item = item.replace("TRESemmu00e9 ", "")

    item = item.replace("u00a0", "")
    if (item[0] == " "):
        item = item[1:]

    if (item.find("barbecue") != -1): item = "bbq"
    if (item.find("bbq") != -1): item = "bbq"
    if (item.find("parking") != -1): item = "parking"
    if (item.find("conditioner") != -1): item = "conditioner"
    if (item.find("wifi") != -1): item = "wifi"
    if (item.find("shampoo") != -1): item = "shampoo"
    if (item.find("body soap") != -1): item = "body soap"
    if (item.find("refrigerator") != -1): item = "refrigerator"
    if (item.find("pool") != -1): item = "pool"
    if (item.find("garden") != -1): item = "garden"
    if (item.find("oven") != -1): item = "oven"
    if (item.find("dryer") != -1 and item != "hair dryer"): item = "dryer"
    if (item.find("sound system") != -1): item = "sound system"
    if (item.find("washer") != -1): item = "washer"
    if (item.find("heating") != -1): item = "heating"
    if (item.find("children books and toys") != -1): item = "children"
    if (item.find("clothing") != -1) : item = "clothing"
    if (item.find("years old") != -1): item = "children"
    if (item.find("carport") != -1): item = "parking"
    if (item.find("game") != -1 or item.find("ps") != -1 or item.find("wii") != -1): item = "game"
    if (item.find("stove") != -1): item = "stove"
    if (item.find("coffee") != -1): item = "coffee"
    if (item.find("closet") != -1): item = "clothing"
    if (item.find("gym") != -1): item = "gym"
    if (item.find("childrenu") != -1): item = "children"
    if (item.find("chair") != -1): item = "chair"

    if (item.find("air conditioning") != -1): item = "air conditioning"
    if (item.find("wardrobe") != -1): item = "clothing"
    if (item.find("balcony") != -1): item = "balcony"
    if (item.find("dresser") != -1): item = "clothing"
    if (item.find("kettle") != -1): item = "kettle"
    if (item.find("hot tub") != -1) : item = "hot tub"
    if (item.find("tv") != -1): item = "tv"


    return item

amenities = listings["amenities"].str.split(",",expand=True)
print(amenities.head())
amenities = amenities.replace(regex=["[^\w\s]"], value="")
for i in range(amenities.shape[0]):
    for j in range(amenities.shape[1]):
        amenities.iloc[i, j] = process_amenities(amenities.iloc[i, j])

amenities_list = [amenities[item].unique().tolist() for item in amenities.columns.values]
amenities_list = set(list(chain.from_iterable(amenities_list)))
amenities_list.remove('')
amenities_list.remove(None)
for i in amenities_list:
    print(i)
print(len(amenities_list))
for items in amenities_list:
    listings["amen-"+items] = listings["amenities"].apply(lambda x: 1 if items in x else 0)

listings.drop(columns="amenities",inplace=True)
drop_10 = [col for col in listings.columns if ((col not in keys) and (listings[col].sum() < 2))]

listings = listings.drop(columns=drop_10, axis=1)

uniques = listings["host_response_time"].unique().tolist()
listings['host_response_time'] = listings['host_response_time'].replace(to_replace=uniques, value=[1, 2, 0, 3, 4])
# host response rate
listings['host_response_rate'] = listings['host_response_rate'].replace(regex=["%"], value="").astype(np.float16) / 100.
listings['host_response_rate'] = listings['host_response_rate'].fillna(listings['host_response_rate'].mean())
listings['host_response_time'] = listings['host_response_time'].fillna(listings['host_response_time'].mean())
listings['host_acceptance_rate'] = listings['host_acceptance_rate'].replace(regex=["%"], value="").astype(np.float16) / 100.0
listings['host_acceptance_rate'] = listings['host_acceptance_rate'].fillna(listings['host_acceptance_rate'].mean())
listings['host_is_superhost'] = listings['host_is_superhost'].replace(to_replace=['f', 't'], value=[0, 1])
listings["price"] = listings["price"].replace({"\$": "", ",": ""}, regex=True)
listings["price"] = listings["price"].astype(float)

uniques = listings["host_neighbourhood"].unique().tolist()
listings['host_neighbourhood'] = listings['host_neighbourhood'].replace(to_replace=uniques, value=[x for x in range(len(uniques))])
listings['host_has_profile_pic'] = listings['host_has_profile_pic'].replace(to_replace=['f', 't'], value=[0, 1])
listings['host_identity_verified'] = listings['host_identity_verified'].replace(to_replace=['f', 't'], value=[0, 1])
uniques = listings["property_type"].unique().tolist()
listings['property_type'] = listings['property_type'].replace(to_replace=uniques, value=[x for x in range(len(uniques))])
uniques = listings["room_type"].unique().tolist()
listings['room_type'] = listings['room_type'].replace(to_replace=uniques, value=[x for x in range(len(uniques))])
uniques = listings["bathrooms_text"].unique().tolist()
listings['bathrooms_text'] = listings['bathrooms_text'].replace(to_replace=uniques, value=[x for x in range(len(uniques))])
listings['bedrooms'] = listings['bedrooms'].fillna(listings['bedrooms'].mean())
listings['beds'] = listings['beds'].fillna(listings['beds'].mean())

print(listings.keys())


stemmer = PorterStemmer()

listings['description'] = listings['description'].fillna("")
listings['description'] = listings['description'].apply(lambda x : re.sub(r"\(.*?\)|\<.*?\>|\[.*?\]", "", string=x))

# remove puctuation marks
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
for punc in punctuation:
    listings['description'] = listings['description'].apply(lambda x : x.replace(punc, " "))
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
listings['description'] = listings['description'].apply(lambda x : get_solo(x))
print(listings.head())
print("removed puctuation marks")
# convert text to lowercase
listings['description'] = listings['description'].str.lower()
print(listings['description'].head())
print('converted to lowercase')
#remove numbers
listings['description'] = listings['description'].str.replace("[0-9]", " ")
print(listings['description'].head())
print("removed numbers")
# remove whitespaces
listings['description'] = listings['description'].apply(lambda x:' '.join(x.split()))
print(listings['description'].head())
print("removed whitespaces")
listings['description'] = listings['description'].apply(lambda x: ' '.join(word_tokenize(x)))
print(listings['description'].head())
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
listings['description'] = stemer(listings['description'])
# # lemmatized
listings['description'] = lemmatization(listings['description'])
print(listings['description'].head())
print("finish lemmatized")
print("finish preprocessing")



vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.9, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(listings['description'])
print(len(vectorizer.get_feature_names()))

print(vectorizer.get_feature_names())
X = X.toarray()

trainDataVecs = np.array(X)
print(trainDataVecs.shape)
# for items in amenities_list:
#     listings[items] = listings["amenities"].apply(lambda x: 1 if items in x else 0)
for i, x in enumerate(vectorizer.get_feature_names()):
    listings["tf-des-" + x] = trainDataVecs[:, i]
    # train_data_vecs_listings = pd.DataFrame(data=trainDataVecs, columns=["tf-des-" + x for x in vectorizer.get_feature_names()])
print(listings.keys())

# print(train_data_vecs_listings.head())
#
# train_data_vecs_listings.to_csv("../data/reviews_tf-ilistings.csv", index=False)

listings['neighborhood_overview'] = listings['neighborhood_overview'].fillna("")
listings['neighborhood_overview'] = listings['neighborhood_overview'].apply(lambda x : re.sub(r"\(.*?\)|\<.*?\>|\[.*?\]", "", string=x))

# remove puctuation marks
punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'
for punc in punctuation:
    listings['neighborhood_overview'] = listings['neighborhood_overview'].apply(lambda x : x.replace(punc, " "))
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
listings['neighborhood_overview'] = listings['neighborhood_overview'].apply(lambda x : get_solo(x))
print(listings.head())
print("removed puctuation marks")
# convert text to lowercase
listings['neighborhood_overview'] = listings['neighborhood_overview'].str.lower()
print(listings['neighborhood_overview'].head())
print('converted to lowercase')
#remove numbers
listings['neighborhood_overview'] = listings['neighborhood_overview'].str.replace("[0-9]", " ")
print(listings['neighborhood_overview'].head())
print("removed numbers")
# remove whitespaces
listings['neighborhood_overview'] = listings['neighborhood_overview'].apply(lambda x:' '.join(x.split()))
print(listings['neighborhood_overview'].head())
print("removed whitespaces")
listings['neighborhood_overview'] = listings['neighborhood_overview'].apply(lambda x: ' '.join(word_tokenize(x)))
print(listings['neighborhood_overview'].head())
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
listings['neighborhood_overview'] = stemer(listings['neighborhood_overview'])
# # lemmatized
listings['neighborhood_overview'] = lemmatization(listings['neighborhood_overview'])
print(listings['neighborhood_overview'].head())
print("finish lemmatized")
print("finish preprocessing")


vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.9, stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(listings['neighborhood_overview'])
print(len(vectorizer.get_feature_names()))

print(vectorizer.get_feature_names())
X = X.toarray()

trainDataVecs = np.array(X)
print(trainDataVecs.shape)
# for items in amenities_list:
#     listings[items] = listings["amenities"].apply(lambda x: 1 if items in x else 0)
for i, x in enumerate(vectorizer.get_feature_names()):
    listings["tf-nei-" + x] = trainDataVecs[:, i]

print(listings.keys())
listings.drop(columns="description",inplace=True)
listings.drop(columns="neighborhood_overview",inplace=True)
target = ['review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value']
listings = listings.dropna(subset=target)
for key in listings.keys():
    if key == 'id' or key in target:
        continue
    listings[key]=(listings[key]-listings[key].min())/(listings[key].max()-listings[key].min())


listings.to_csv("../data/listings_features.csv", index=False)




