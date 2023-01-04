
import pandas as pd
from tqdm import tqdm
import time
from googletrans import Translator
from langdetect import detect, LangDetectException

reviews = pd.read_csv("../data/reviews_preprocessing_translated_4.csv", dtype={'comments': str})
f = open("../data/translated.txt", "w")
def translatefunc(texts):
    output = []
    translator = Translator(service_urls=['translate.google.com', ])  # 如果可以上外网，还可添加 'translate.google.com' 等
    translator.raise_Exception = True
    all_false = 0
    all_ = 0
    for i in tqdm(texts):
        if i == '':
            output.append(" ")
            continue
        try:
            lang = detect(i)
            if lang != 'en':
                all_ += 1
                f.write("original = " + i + "\n")
                try:
                    print(lang)
                    print("original = ", i)
                    word = translator.translate(i, dest='en').text
                    output.append(word)
                    print("tranlated = ", word)
                except:
                    time.sleep(10)
                    output.append(str(i))
                    all_false += 1
                    f.write("Error occured\n")
                    print(f"Error occured {all_false}, {all_}")
            else:
                output.append(str(i))
        except:
            output.append(str(i))
    return output
# f.close()
reviews['comments'] = translatefunc(reviews['comments'])
reviews.to_csv("../data/reviews_translated.csv", index=False)
f.close()