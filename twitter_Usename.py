# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 21:10:38 2020

@author: malat
"""
import pandas as pd

import GetOldTweets3 as got
tweetCriteria = got.manager.TweetCriteria().setUsername("narendramodi")\
                                           .setMaxTweets(20)
tweet = got.manager.TweetManager.getTweets(tweetCriteria)


df = pd.DataFrame(t.__dict__ for t in tweet)
