#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 14:45:19 2023

@author: vamsik
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('./Documents/Vamsi/DS/Kaggle/train.csv')
df= df.drop(['Name', 'PassengerId', 'Ticket','Embarked','Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())

encoded= pd.get_dummies(df[['Sex']], drop_first=True)
df = pd.concat([df, encoded], axis=1)
df= df.drop(['Sex'], axis=1)


y_train = df['Survived']
X_train = df.drop('Survived', axis=1)

df1 = pd.read_csv('./Documents/Vamsi/DS/Kaggle/test.csv')
df1= df1.drop(['Name', 'PassengerId', 'Ticket','Embarked','Cabin'], axis=1)
df1['Age'] = df1['Age'].fillna(df['Age'].median())
df1['Fare'] = df1['Fare'].fillna(df['Fare'].median())

encoded1 = pd.get_dummies(df1[['Sex']], drop_first=True)
df1 = pd.concat([df1, encoded1], axis=1)
df1 = df1.drop(['Sex'], axis=1)

X_test = test_df


#splitting train data in 2 parts

from sklearn.model_selection import train_test_split
xtraining,xvalid,ytraining,yvalid = train_test_split(X,y,test_size=0.2)

#importing the packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

#specifying the initial learners
model1 = RandomForestClassifier(random_state=42,n_estimators=200,max_depth=7 )
model2 = AdaBoostClassifier(random_state=1,n_estimators=750,learning_rate=0.7)
model3 = GradientBoostingClassifier(learning_rate=0.008,random_state=1,n_estimators=5000)
model4 = xgb.XGBClassifier(random_state=1,learning_rate=0.05,n_estimators = 500)

#training the initial learners
model1.fit(xtraining,ytraining)
model2.fit(xtraining,ytraining)
model3.fit(xtraining,ytraining)
model4.fit(xtraining,ytraining)

#making predictions for the validation data
preds1 = model1.predict(xvalid)
preds2 = model2.predict(xvalid)
preds3 = model3.predict(xvalid)
preds4 = model4.predict(xvalid)

#making predictions for the test data
test_preds1 = model1.predict(X_test)
test_preds2 = model2.predict(X_test)
test_preds3 = model3.predict(X_test)
test_preds4 = model4.predict(X_test)



#making a new dataset for training our final model by stacking the predictions on the validation data
train_stack = np.column_stack((preds1,preds2,preds3,preds4))

#making the final test set for our final model by stacking the predictions on the test data
test_stack = np.column_stack((test_preds1,test_preds2,test_preds3,test_preds4))


final_model = RandomForestClassifier()

#training the final model on the stacked predictions
final_model.fit(train_stack,yvalid)

final_predictions = final_model.predict(test_stack)

test_passenger_ids['Transported'] = final_predictions
test_passenger_ids['Transported'] = test_passenger_ids['Transported'].astype(bool)

test_passenger_ids.to_csv('./Documents/Vamsi/DS/Kaggle/Spaceship Titanic/submission_11.csv',index = False)
