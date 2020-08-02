# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:41:30 2020

@author: malat
"""


import pandas as pd

import numpy as np

data = pd.read_csv("new.csv")

data["sentiment"].value_counts()

blanks = []

for i, r, s in df.itertuples():
    if type(r) == str:
        if r.isspace():
            blanks.append(i)

print(len(blanks), blanks)

from sklearn.model_selection import train_test_split
X = data['review']
y = data["sentiment"]

x_train, x_test,y_train, y_test = train_test_split(X,y, test_size = 0.01)


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()

x_train_tf = tf.fit_transform(x_train)
x_train_tf.shape


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(x_train_tf,y_train)

from  sklearn.pipeline import Pipeline
text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', MultinomialNB()),
])

text_clf.fit(x_train,y_train)


p = text_clf.predict(x_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,p)
cm


from sklearn.metrics import accuracy_score
accuracy_score(y_test,p)


from sklearn.svm import LinearSVC
test_clf_1 = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
                     ])
test_clf_1.fit(x_train,y_train)

p2 = test_clf_1.predict(x_test)

confusion_matrix(y_test,p2)
accuracy_score(y_test,p2)

test = pd.read_csv("penguin keerthi suresh.csv")

y = test["Review"]
x = test["text"]
new1 = text_clf.predict(x)
new2 = test_clf_1.predict(x)


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
confusion_matrix(y,new1)
confusion_matrix(y,new2)

accuracy_score(y,w1)
accuracy_score(y,new2)


import pickle

filename = 'model.sav'
pickle.dump(text_clf, open(filename, 'wb'))

import pickle

m = pickle.load(open("model.sav", 'rb'))
w1 = m.predict(x)


TfidfVectorizer()




