# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:38:41 2020

@author: malat
"""

# importing the libraries
import time

import pandas as pd

import numpy as np
 
import matplotlib.pyplot as plt

import seaborn as sns
# importing the dataset

data = pd.read_csv("new.csv")

#cleaning the text
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

cleanup_nums = {"sentiment": {"positive": 1, "negative": 0}}
data.replace(cleanup_nums, inplace=True)

corpus = []
for i in range(0,5000):
    review = re.sub('[^a-zA-Z]', ' ', data['review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = None,ngram_range = (1,2), max_df = 0.80, min_df = 2)
X = cv.fit_transform(corpus)
y = data.iloc[:, 1].values


#Spiliting the dataset into train and test
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.2)

#tarining the naive bayes model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,Y_train)

#predicting the test results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)

























