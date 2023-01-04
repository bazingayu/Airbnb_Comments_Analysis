from tqdm import tqdm
from googletrans import Translator
import time
import numpy as np
import pandas as pd
f1 = open("data.txt", "r")
data1 = f1.readlines()
len1 = len(data1)
print(len1)
f = open("data.txt", "a+")
data = f.readlines()
# len1 = len(data)
# print(len1)
translator = Translator(service_urls=['translate.google.com',])# 如果可以上外网，还可添加 'translate.google.com' 等
reviews = pd.read_csv("./data/reviews.csv", dtype={'comments': str})
def translatefunc(texts):
    translator = Translator(service_urls=['translate.google.com', ])  # 如果可以上外网，还可添加 'translate.google.com' 等
    translator.raise_Exception = True
    for index, i in tqdm(enumerate(texts)):
        if index < len1:
            print(index)
            continue
        s = translator.translate(i, dest='en')
        print(s)
        # print("after_translate", s.text)
        f.writelines(str(index) + " " + s.text)
        # f.writelines(translator.translate(i, dest='en'))

translatefunc(reviews['comments'])
