import pandas as pd
from itertools import chain

df = pd.read_csv("../data/listings.csv")
keys = df.keys()

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
    if (item.find("tv") != -1): item = "tv"
    if (item.find("parking") != -1): item = "parking"
    if (item.find("conditioner") != -1): item = "conditioner"
    if (item.find("wifi") != -1): item = "wifi"
    if (item.find("shampoo") != -1): item = "shampoo"
    if (item.find("body soap") != -1): item = "body soap"
    if (item.find("refrigerator") != -1): item = "refrigerator"
    if (item.find("pool") != -1): item = "pool"
    if (item.find("garden") != -1): item = "garden"
    if (item.find("oven") != -1): item = "oven"
    if (item.find("sound system") != -1): item = "sound system"
    if (item.find("washer") != -1): item = "washer"
    if (item.find("heating") != -1): item = "heating"
    if (item.find("children books and toys") != -1): item = "children books and toys"
    if (item.find("carport") != -1): item = "parking"
    if (item.find("game") != -1 or item.find("ps") != -1 or item.find("wii") != -1): item = "game"
    if (item.find("stove") != -1): item = "stove"
    if (item.find("coffee") != -1): item = "coffee"
    if (item.find("closet") != -1): item = "closet"
    return item

amenities = df["amenities"].str.split(",",expand=True)
print(amenities.head())
amenities = amenities.replace(regex=["[^\w\s]"], value="")
for i in range(amenities.shape[0]):
    for j in range(amenities.shape[1]):
        amenities.iloc[i, j] = process_amenities(amenities.iloc[i, j])

amenities_list = [amenities[item].unique().tolist() for item in amenities.columns.values]
amenities_list = set(list(chain.from_iterable(amenities_list)))
amenities_list.remove('')
amenities_list.remove(None)
print(len(amenities_list))
for items in amenities_list:
    df[items] = df["amenities"].apply(lambda x: 1 if items in x else 0)

df.drop(columns="amenities",inplace=True)
# df.to_csv("../data/amentities1.csv", index=False)
print(len(df.keys()))
drop_10 = [col for col in df.columns if ((col not in keys) and (df[col].sum() < 5))]

df = df.drop(columns=drop_10, axis=1)
print(len(df.keys()) - len(keys))
# df.to_csv("../data/listings_cleaned_processed_amenities.csv", index=False)