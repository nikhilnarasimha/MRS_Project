# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 22:03:09 2020

@author: malat


"""
income = input("search :" )

import time

import pandas as pd

import GetOldTweets3 as got

import numpy as np

import re

import pickle 

import nltk

from nltk.tokenize import word_tokenize

from nltk.stem.snowball import SnowballStemmer

s_stemmer = SnowballStemmer(language='english')

from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline

start = time.time()

income = income.split(",")


#nltk.download("stopwords")

#nltk.download('wordnet')

data = pd.DataFrame(columns=['username', 'to', 'text', 'retweets', 'favorites', 'replies', 'id',
       'permalink', 'author_id', 'date', 'formatted_date', 'hashtags',
       'mentions', 'geo', 'urls'])
  

n = income

print(f'Your Search Query : {n}')
print("please wait....... we will let you know the rating")

for i in n:
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(i)\
                                           .setMaxTweets(500)
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)

    df = pd.DataFrame(t.__dict__ for t in tweet)

    data = data.append(df)

data.drop_duplicates(subset ="text", inplace = True)
"""
name = n[0]

name = name + ".csv"

df.to_csv(name, index=False)
"""


#lag
#from googletrans import Translator
#tra =   Translator()

#row = ["ela undi","super undi","ala undi"]

#tra.translate("ela undi",dest = "en").text
#for i in row:
        #print(tra.translate(i).text)

raw = data[["text"]]

#Removing unwanted Chars

def cleantext(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) #Remove @mentions
    text = re.sub(r'#', '',text)
    text = re.sub(r'https?:\/\/\S+','',text)#Removing # from tweets
    text = re.sub(r'  +', ' ', text) #removing extra spaces  
    return text

raw_2 = []

for i, j in raw.itertuples():
    cl = cleantext(j)
    raw_2.append(cl)
    
# Removing the http Tweets from the data
'''for t in raw_2:
    if re.search("http",t):
        raw_2.remove(t)'''


df = pd.DataFrame(raw_2, columns = ["text"])

# loading the contraction dict
f = open('contraction.pkl', 'rb')
contraction = pickle.load(f)


#changing the contration words
for i, j in df.itertuples():
    for word in j.split():
        if word.lower() in contraction:
           m = j.replace(word, contraction[word.lower()])
           df.xs(i)["text"] = m

#changing the contration words---2
           
for i, j in df.itertuples():
    for word in j.split():
        if word.lower() in contraction:
           m = j.replace(word, contraction[word.lower()])
           df.xs(i)["text"] = m

#all
stopwords = pd.read_csv("stopwords.csv")
s = stopwords["0"]
sw = set(s)

#removing the search
for i in n:
    stops = word_tokenize(i)
    stops = [word for word in stops if word.isalpha()]
    stops = [word.lower() for word in stops]
    stops = ' '.join(stops)
    sw.add(i)
   
for i, j in df.itertuples():
    tokens = word_tokenize(j)
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if not word in sw]
    #words = [lemmatizer.lemmatize(word) for word in tokens]
    words = [s_stemmer.stem(word) for word in tokens]
    sen = ' '.join(words)
    df.loc[i,"text"] = sen
  
x_test = df["text"]

model = pickle.load(open("model.sav","rb"))


y_pred = model.predict(x_test)


y = pd.DataFrame(y_pred, columns = ["result"] )
n_values = y["result"].value_counts()
pos = n_values[1]


total = len(x_test)
score = (pos / total)
score
per = round(score*100,1)
likage = f'{per}%'

print(likage)

end = time.time()

print(end - start)








