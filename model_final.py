# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:37:20 2020

@author: malat
"""


import pandas as pd

import numpy as np

from collections import Counter

x = df["review"]
y = data["sentiment"]


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv_fit = cv.fit_transform(x)
cv_fit.shape

# word counts
vocab = list(cv.get_feature_names())
counts = cv_fit.sum(axis = 0).A1

freq_distribution = Counter(dict(zip(vocab, counts)))
print (freq_distribution.most_common(100))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

text_clf_NB = Pipeline([("cv", CountVectorizer(ngram_range= (1,2))),
                         ("tf",TfidfTransformer(use_idf= False)),
                     ("clf", MultinomialNB(alpha= 0.01 )),
 ])

text_clf_NB.fit(x,y)

text_clf_SVM = Pipeline([("cv", CountVectorizer(ngram_range= (1,2))),
                         ("tf",TfidfTransformer()),
                     ("clf", SGDClassifier()),
                     ])


text_clf_SVM.fit(x,y)


text_clf_Lin = Pipeline([("cv", CountVectorizer()),
                         ("tf",TfidfTransformer()),
                     ("clf", LinearSVC()),
 ])

text_clf_Lin.fit(x,y)

text_clf_XG = Pipeline([("cv", CountVectorizer()),
                         ("tf",TfidfTransformer()),
                     ("clf", XGBClassifier()),
])

text_clf_XG.fit(x,y)


test = pd.read_csv("breathe.csv")

x_test_b = df["text"]
y_test_b = test["review"]


y_pred_SVM = text_clf_SVM.predict(x_test_b)

y_pred_NB = text_clf_NB.predict(x_test_b)


df["svm"] = y_pred_SVM
test["XG_1"] = y_pred_NB
test["XG"] = y2
test["svm"].value_counts()
test["XG_1"].value_counts()
test["review"].value_counts()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

print(f'SVM : {accuracy_score(y_test_b,y_pred_SVM)}')
print(f'Lin : {accuracy_score(y_test_b,y2)}')
print(f'Nb : {accuracy_score(y_test_b,y_pred_NB)}')

print(f'SVM : {recall_score(y_test_b,y_pred_SVM)}')
print(f'Lin : {recall_score(y_test_b,y_pred_Lin)}')
print(f'Nb : {recall_score(y_test_b,y_pred_NB)}')

print(f'SVM : {precision_score(y_test_b,y_pred_SVM)}')
print(f'Lin : {precision_score(y_test_b,y_pred_Lin)}')
print(f'Nb : {precision_score(y_test_p,y_pred_NB)}')

print(f'SVM : {f1_score(y_test_b,y_pred_SVM)}')
print(f'Lin : {f1_score(y_test_b,y_pred_Lin)}')
print(f'Nb : {f1_score(y_test_p,y_pred_NB)}')


cm = confusion_matrix(y_test_b,y_pred_NB)
plot.imshow(cm, cmap = 'binary')


df_cm = pd.DataFrame(cm)

sn.heatmap(df_cm, annot=True)
import seaborn as sn

y_pred_Lin


from sklearn.metrics import recall_score
recall_score(y_test,y_pred)


#Grid SearchCV
from sklearn.model_selection import GridSearchCV
parameters = {'cv__ngram_range': [(1,3),(1,2)],
              'cv__max_df': (1, 0.90),
              'cv__min_df': (1, 5),
              'tf__use_idf': (True, False),
              'clf__alpha':(1e-2, 1e-3),
}

GS_clf_nb = GridSearchCV(text_clf_NB,parameters,n_jobs=-1, cv = 3)
gs_clf_n = GS_clf_nb.fit(x,y)

GS_clf_SVM = GridSearchCV(text_clf_SVM,parameters,n_jobs=-1, cv = 3)
gs_clf_s = GS_clf_SVM.fit(x,y)


gs_clf_n.best_score_

import xgboost 
from xgboost import XGBClassifier

GS_clf_S.be
gs_clf_s.best_score_
gs_clf_s.best_params_
y_pred_SVM_1 = gs_clf_s.predict(x_test_b)


#pickle
import pickle

filename = 'model.sav'
pickle.dump(text_clf, open(filename, 'wb'))

import pickle

model = pickle.load(open("model.sav", 'rb'))


y_pred = model.predict(x_test)


