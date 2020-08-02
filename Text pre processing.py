# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:02:35 2020

@author: malat
"""

#importing lib
import pandas as pd

import pickle 

import nltk

from nltk.tokenize import word_tokenize

import re

from nltk.stem.snowball import SnowballStemmer
s_stemmer = SnowballStemmer(language='english')
#nltk.download("stopwords")

#nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

#importing the data
data = pd.read_csv("IMDB Dataset.csv")

df = raw[["text"]]

# loading the contraction dict
f = open('contraction.pkl', 'rb')
contraction = pickle.load(f)



#changing the contration words
for i, j in df.itertuples():
    for word in j.split():
        if word.lower() in contraction:
           m = j.replace(word, contraction[word.lower()])
           df.xs(i)["text"] = m


for i, j in df.itertuples():
    for word in j.split():
        if word.lower() in contraction:
            m = j.replace(word, contraction[word.lower()])
            print(word)


#removing pun

#for i, j in df.itertuples():
    #tokens = word_tokenize(j)
    #words = [word for word in tokens if word.isalpha()]
    #sen = ' '.join(words)
    #df.xs(i)["text"] = sen
    
#lower case
#for i, j in df.itertuples():
    #tokens = word_tokenize(j)
    #words = [word.lower() for word in tokens]
    #sen = ' '.join(words)
    #df.xs(i)["text"] = sen


#Stopwords
#stopwords = pd.read_csv("stopwords.csv")
#s = stopwords["0"]
#sw = set(s)

#for i, j in df.itertuples():
    #tokens = word_tokenize(j)
    #words = [word for word in tokens if not word in sw]
    #sen = ' '.join(words)
    #df.xs(i)["text"] = sen


#lemma
#lemmatizer = WordNetLemmatizer() 
#for i, j in df.itertuples():
    #tokens = word_tokenize(j)
    #words = [lemmatizer.lemmatize(word) for word in tokens]
    #sen = ' '.join(words)
    #df.xs(i)["text"] = sen  

#all
stopwords = pd.read_csv("stopwords.csv")
s = stopwords["0"]
sw = set(s)
  
for i, j in df.itertuples():
    tokens = word_tokenize(j)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if not word in sw]
    #words = [lemmatizer.lemmatize(word) for word in tokens]
    words = [s_stemmer.stem(word) for word in tokens]
    sen = ' '.join(words)
    df.xs(i)["text"] = sen


#converting cat to num
cleanup_nums = {"review":{0: 0, 4: 1}}
y.replace(cleanup_nums, inplace= True)


y = data[["0"]]
y["review"].value_counts()





y = y.rename(columns = {"0":"review"})





