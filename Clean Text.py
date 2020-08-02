# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 14:13:33 2020

@author: malat
"""


import pandas as pd

import numpy as np

import re

data = pd.read_csv("Onward movie.csv")

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

data1 = data[["text"]]

# Removing the http Tweets from the data


http = []

for i, t in raw.itertuples():
    if re.search("http",t):
        http.append(i)
        
len(http)


junk = []

for i in http:
    j = raw["text"][i]
    junk.append(j)


raw.drop(http, inplace=True)

raw.to_csv("onwards.csv", index= False)

for i, j in raw.itertuples():
    cl = cleantext(j)
    raw.xs(i)["text"] = cl
    
    

income = ["ram", "ram_krishna"]

def WS(income):
    text = []
    for i in  income:
        a= i.replace("_"," ")
        text.append(a)
    return text
    

def twittersearch(text):
    data = pd.DataFrame(columns=['username', 'to', 'text', 'retweets', 'favorites', 'replies', 'id',
       'permalink', 'author_id', 'date', 'formatted_date', 'hashtags',
       'mentions', 'geo', 'urls'])
    for i in text:
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(i)\
                                           .setMaxTweets(5)
        tweet = got.manager.TweetManager.getTweets(tweetCriteria)

        df = pd.DataFrame(t.__dict__ for t in tweet)

        data = data.append(df)
        
    data.drop_duplicates(subset ="text", inplace = True)
    
    return data

im = ["modi", "covid"]

data = twittersearch(im)







