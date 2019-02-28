# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:43:06 2019

@author: AKarsh K M
"""

import numpy as np
import pandas as pd
from sklearn import tree 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("Path to dataset -- csv file",na_values = ['?'], names = ['BI-RADS','age','shape','margin','density','severity'])

data.dropna(inplace=True)

all_features = data[['age','shape','margin','density']].values
all_classes = data['severity'].values
feature_names = ['age','shape','margin','density']

scaler = preprocessing.StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)


x_train,x_test,y_train,y_test =  train_test_split(all_features_scaled,all_classes, random_state=0)

forest = RandomForestClassifier(n_estimators = 100,random_state=0)
forest.fit(x_train,y_train)

print("train accuracy", forest.score(x_train,y_train))
print("test accuracy",forest.score(x_test,y_test))
