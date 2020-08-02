# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 19:52:53 2020

@author: malat
"""

# vader
import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sa = SentimentIntensityAnalyzer()

sa.polarity_scores(x[0])

y_pred = x.apply(lambda x: sa.polarity_scores(x))

score = y_pred.apply(lambda y: y["compound"])

y1 = score.apply(lambda s: 1 if s >= 0.05 else 0)

confusion_matrix(y,y1)

accuracy_score(y,y1)

# Textblob

from textblob import TextBlob

TextBlob.sentiment.polarity(x[0])
x = df["text"]

ps = x.apply(lambda x: TextBlob(x).sentiment)

score1 = ps.apply(lambda y: y.polarity)


y2 = score1.apply(lambda s: 1 if s >= 0.05 else 0)


confusion_matrix(y,y2)

accuracy_score(y,y2)


#Watson IBM
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, SentimentOptions, CategoriesOptions


natural_language_understanding = NaturalLanguageUnderstandingV1(version='2019-07-12',
    iam_apikey='tfC7EkZCNwvqRjKNMQuGddwiki0wMCBViolplrSpm3Yy', # Use your API key here
    url='https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/7739f5d0-e7a6-49b9-b489-8d27eeee9024' # paste the url here
    )

import ibm_watson
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, SemanticRolesOptions

authenticator = IAMAuthenticator('tfC7EkZCNwvqRjKNMQuGddwiki0wMCBViolplrSpm3Yy')
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2019-07-12',
    authenticator=authenticator
)

natural_language_understanding.set_service_url('https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com/instances/7739f5d0-e7a6-49b9-b489-8d27eeee9024')



response = natural_language_understanding.analyze( text = "movie was good",
                                                  features=Features(sentiment=SentimentOptions()).get_result()


response = natural_language_understanding.analyze(
    text = "movie good",
    features=Features(sentiment=SentimentOptions())).get_result()

res = response.get('sentiment').get('document').get('score')
print(res)

ibm_score = data1.apply(lambda a: Sentiment_score(a))



input_text = "m"

def Sentiment_score(input_text): 
    # Input text can be sentence, paragraph or document
    response = natural_language_understanding.analyze (
    text = input_text,
    features = Features(sentiment=SentimentOptions())).get_result()
    # From the response extract score which is between -1 to 1
    res = response.get('sentiment').get('document').get('score')

ibm_score = []

for i, j in data1.itertuples():
    s = Sentiment_score(j)
    print(s)
    print(j)
    













