import requests
from keras.preprocessing.text import Tokenizer
import pickle
from keras.utils import to_categorical, pad_sequences
import numpy as np
import torch
import math
import pandas as pd
import csv

#Gets image given url in dataset and downloads image
def get_image(url, UID):
    response = requests.get(url)

    with open(f"./images/{UID}.jpg", "wb") as f:
        f.write(response.content)

#Tokenizes all the descriptions given
def tokenizer(captions):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    pickle.dump(tokenizer, open('tokenizer.p', 'wb'))
    return tokenizer

#Cleans the descriptions, removes excess/uneccessary words
def get_desc(des, about, id):
    review_list = []
    f = open('edited.csv', 'w')
    writer= csv.writer(f)

    for d in range(len(des)):
        try:
            if math.isnan(des[d]):
                a = about[d]

                try:
                    math.isnan(a)
                    review_list.append("<end>")
                    writer.writerow([id[d], "<end>"])
                    continue
                except:
                    ex1 = a.split("number. |")

                    review_list.append(ex1[-1] + " <end>")
                    writer.writerow([id[d], ex1[-1] + " <end>"])
                    continue
        except:
            ex1 = des[d].split("default")

            if len(ex1) > 1:
                ex1 = ex1[1] #anything after default
            else:
                ex1= ex1[0]

            ex2 = ex1.split("Ship it! |")

            if len(ex2) > 1:
                ex3 = ex2[-1]
            else:
                ex3= ex2[0]
            
            ex4 = ex3.split("|")

            if len(ex4) > 1:
                ex5 = ex4[1].strip()

                if ex5[0].isnumeric():
                    final= ex4[0]
                else:
                    final = ex5
            else:
                final= ex4[0].strip()
        
            sent = final + " <end>"
            review_list.append(sent)
            writer.writerow([id[d], sent])

    return pd.Series(review_list)

#Gets the tokenized values assigned to each caption given the tokenizer word set
def get_tokens(token, captions):
    new_list = []

    for caps in captions:
        new_list.append(token.texts_to_sequences([caps])[0])

    return new_list