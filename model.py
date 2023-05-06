# -*- utf-8 -*-
# @Author: Huiling Jia
# @Software: PyCharm
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import os




def run_NaiveBayes_model(X_test,y_test,X_train=None,y_train=None):
    if X_train is None:
        with open('nb_model.pickle', 'rb') as handle:
            model = pickle.load(handle)
    else:
        model = GaussianNB()
        model.fit(X_train, y_train)
        cscore = cross_val_score(estimator=model, X=X_train, y=y_train, cv=5)
        print(f'cross validation score: {cscore}')
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy score: ', score)
    cf_matrix = confusion_matrix(y_test, y_pred)
    print(cf_matrix)


if __name__ == '__main__':
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=40)
    X_test = pd.read_csv('./Testdata.csv').values
    y_test = pd.read_csv('./Testlabel.csv').values
    run_NaiveBayes_model(X_test,y_test)
