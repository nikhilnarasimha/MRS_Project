# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:13:30 2020

@author: malat
"""


import pandas as pd

import GetOldTweets3 as got

data = pd.DataFrame(columns=['username', 'to', 'text', 'retweets', 'favorites', 'replies', 'id',
       'permalink', 'author_id', 'date', 'formatted_date', 'hashtags',
       'mentions', 'geo', 'urls'])

n = ["Onward movie", '#breathemovie',"#peNium movIe" ]

import time

start = time.time()
for i in n:
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(i)\
                                           .setMaxTweets(5000)
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)

    df = pd.DataFrame(t.__dict__ for t in tweet)

    data = data.append(df)


name = n[0]

name = name + ".csv"

df.to_csv(name, index=False)

end = time.time()

print(end - start)

