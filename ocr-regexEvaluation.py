#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 19:21:02 2020

@author: muntabir
"""


#importing libraries
import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import xgboost
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import SGDClassifier

#creating a dflist with column name: abstract and label 
dfList=[]
colname=['tokens', 'label']
df = pd.read_csv('regex_error-word.csv', encoding = "utf-8", header = None)
dfList.append(df)
concatDf = pd.concat(dfList, axis =0)
concatDf.columns=colname
concatDf.to_csv('regex_error-word_update.csv', index = None, header=True, encoding = 'utf-8')

# glove_file = "/Users/muntabir/Desktop/Graduate-School-ODU/CS895/project/source/glove.840B.300d.txt"
# words = pd.read_table(glove_file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
# def vec(w):
#   return words.loc[w].as_matrix()
  

df1 = pd.read_csv("regex_error-word_update.csv", encoding = "utf-8")
df1 = df1.dropna()
df1 = df1.reset_index(drop=True)
vectorizer = TfidfVectorizer()
#vectorizer = CountVectorizer()
vectorizer.fit(df1['tokens'])
x = vectorizer.transform(df1['tokens'])
y = df1['label'] #target set

train_X, test_X, train_y, test_y = train_test_split(x, y,
                                                    stratify=y, 
                                                   test_size=0.3, random_state = 1)

#Gradient Boosting Classifier
# lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
# for learning_rate in lr_list:
#     gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=1000, max_depth=2, random_state=0)
#     gb_clf.fit(train_X,train_y)
    
#     print("Learning rate: ", learning_rate)
#     print("Accuracy score (training): {0:.3f}".format(gb_clf.score(train_X, train_y)))
#     print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(test_X, test_y)))

# gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=1, max_features=1000, max_depth=2, random_state=0)
# gb_clf2.fit(train_X, train_y)
# predictions = gb_clf2.predict(test_X)

# print("Confusion Matrix:")
# print(confusion_matrix(test_y, predictions))

# print("Classification Report")
# print(classification_report(test_y, predictions))

#XGboost Classifier
model = xgboost.XGBClassifier()
model.fit(train_X, train_y)
y_pred = model.predict(test_X)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(test_y, predictions)
print("\n")
print("########## OCR Regex Train ###################")
print("Train - Gradient Boost Classifier\n")
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Precision:",metrics.precision_score(test_y, predictions))
print("Recall:",metrics.recall_score(test_y, predictions))
print("F1:",metrics.f1_score(test_y, predictions))

# #Support Vector Machine    
# clf = svm.SVC(kernel='linear')
# clf.fit(train_X, train_y)
# y_pred = clf.predict(test_X)
# predictions = [round(value) for value in y_pred]
# accuracy = accuracy_score(test_y, predictions)
# print("\n")
# print("Train - Support Vector Machine\n")
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# print("Precision:",metrics.precision_score(test_y, predictions))
# print("Recall:",metrics.recall_score(test_y, predictions))
# print("F1:",metrics.f1_score(test_y, predictions))

param_grid_ = [ {'alpha': [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]}]
model = GridSearchCV(SGDClassifier(max_iter=2), cv=5, param_grid=param_grid_, scoring='f1_micro', n_jobs=-1, verbose=10)
model.fit(train_X, train_y)
print(model.cv_results_,'\n\n')
print("Best parameters set found on development set:\n") 
print(model.best_params_,'\n')
print("Grid scores on development set:\n") 
means = model.cv_results_['mean_test_score']
stds = model.cv_results_['std_test_score'] 
for mean, std, params in zip(means, stds, model.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params)) 
    print('\n\n')
    print("Detailed classification report:\n") 
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.\n\n")

test_y, pred_y = test_y, model.predict(test_X)
print(confusion_matrix(test_y, pred_y))
print('\n\n')
print(classification_report(test_y, pred_y))