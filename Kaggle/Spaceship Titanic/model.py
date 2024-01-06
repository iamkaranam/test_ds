#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:28:12 2023

@author: vamsik
"""

import os
import pandas as pd
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt



def get_column_distribution(df,col_name):
        
    # Calculate counts and percentages
    counts = df[col_name].value_counts()
    percentages = (counts / len(train_df)) * 100

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({'Counts': counts, 'Percentages': percentages})

    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Bar plot for Counts
    plt.subplot(2, 1, 1)
    ax = sns.barplot(x=plot_data.index, y='Counts', data=plot_data, palette="viridis")
    plt.title('Absolute Counts of Categories')
    plt.ylabel('Counts')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')


    # Pie plot for Percentages
    plt.subplot(2, 1, 2)
    plt.pie(percentages, labels=percentages.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis"))
    plt.title('Percentage of Categories')

    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    
def get_column_distribution_continous_variable(df,col_name):
        
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col_name], bins=20, kde=False, color='blue')
    plt.title('Histogram of {}'.format(col_name))
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    plt.show()
    

train_df = pd.read_csv('./Documents/Vamsi/DS/Kaggle/Spaceship Titanic/train.csv')


#Individual Column Analysis---------

#column names
train_df.columns

# 'PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age',
#        'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
#        'Name', 'Transported'

# PassengerId > We can drop this column safely. It will not affect the output column

train_df = train_df.drop(['PassengerId','Name'],axis = 1)

train_df = train_df.drop(['VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck'],axis = 1)




# HomePlanet

train_df['HomePlanet'].nunique()

Counter(train_df['HomePlanet'])

get_column_distribution(train_df,'HomePlanet')

# CryoSleep

train_df['CryoSleep'].nunique()

Counter(train_df['CryoSleep'])


# Cabin

train_df['Cabin'] = train_df['Cabin'].fillna('NA')

train_df.isna().sum()

train_df['cabin_1'] = train_df['Cabin'].apply(lambda x : x.split('/')[0] )
train_df['cabin_2'] = train_df['Cabin'].apply(lambda x : x.split('/')[1] if len(x.split('/')) == 3 else x )
train_df['cabin_3'] = train_df['Cabin'].apply(lambda x : x.split('/')[2] if len(x.split('/')) == 3 else x )

# cabin_1 
# cabin_1 seems like a category which can be used to check any further relation with the output variable

train_df['cabin_1'].nunique()

Counter(train_df['cabin_1'])

get_column_distribution(train_df,'cabin_1')


# cabin_2
# ignore cabin_2 for now
train_df['cabin_2'].nunique()

train_df = train_df.drop(['cabin_2'],axis = 1)



# cabin_3

train_df['cabin_3'].nunique()

Counter(train_df['cabin_3'])

get_column_distribution(train_df,'cabin_3')

# dropping main column Cabin

train_df = train_df.drop('Cabin',axis = 1)

# Destination

train_df['Destination'].nunique()

Counter(train_df['Destination'])

get_column_distribution(train_df,'Destination')

# Age

train_df['Age'] = train_df['Age'].apply(lambda x : -1 if str(x) == 'NA' else x)

get_column_distribution_continous_variable(train_df,'Age')


# VIP

train_df['VIP'].nunique()

Counter(train_df['VIP'])

get_column_distribution(train_df,'VIP')

# RoomService

train_df['RoomService'] = train_df['RoomService'].fillna(-1)

get_column_distribution_continous_variable(train_df,'RoomService')

# FoodCourt

train_df['FoodCourt'] = train_df['FoodCourt'].fillna(-1)

get_column_distribution_continous_variable(train_df,'FoodCourt')

# ShoppingMall

train_df['ShoppingMall'] = train_df['ShoppingMall'].fillna(-1)

get_column_distribution_continous_variable(train_df,'ShoppingMall')

# Spa

train_df['Spa'] = train_df['Spa'].fillna(-1)

get_column_distribution_continous_variable(train_df,'Spa')

# VRDeck

train_df['VRDeck'] = train_df['VRDeck'].fillna(-1)

get_column_distribution_continous_variable(train_df,'VRDeck')

# Name column can be eliminated
train_df = train_df.drop(['Name'],axis = 1)

# Transported

train_df['Transported'].nunique()

Counter(train_df['Transported'])


#### ------------  train_df data processing --------------
import math

train_df = pd.read_csv('./Documents/Vamsi/DS/Kaggle/Spaceship Titanic/train.csv')

train_df['cabin_1'] = train_df['Cabin'].apply(lambda x : x.split('/')[0] if not pd.isna(x) else x )
train_df['cabin_3'] = train_df['Cabin'].apply(lambda x : x if pd.isna(x) else x.split('/')[2] if len(x.split('/')) == 3 else x )

train_df = train_df.drop('Cabin',axis = 1)


train_df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] = train_df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(0)

train_df['total_spent'] = train_df['RoomService'] + train_df['FoodCourt'] + train_df['ShoppingMall'] + train_df['Spa'] + train_df['VRDeck']

train_df['total_spent'] = train_df['total_spent'] > 0

categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'cabin_1', 'cabin_3','VIP','total_spent']

# Filling missing values with the most frequent value (mode)

for column in categorical_columns:
    mode_value = train_df[column].mode().iloc[0]  # Get the most frequent value
    train_df[column].fillna(mode_value, inplace=True)

continuous_columns = ['Age']

# Filling missing values with the mean

for column in continuous_columns:
    mean_value = train_df[column].mean()
    train_df[column].fillna(mean_value, inplace=True)

# train_df['Age'] = train_df['Age'].apply(lambda x : math.floor(x/10))
    

train_df.isna().sum()

train_df['VIP'] = train_df['VIP'].astype(int)
train_df['Transported'] = train_df['Transported'].astype(int)
train_df['total_spent'] = train_df['total_spent'].astype(int)
train_df['CryoSleep'] = train_df['CryoSleep'].astype(int)

# HomePlanet_dict = {'Earth':1,'Europa':2,'Mars':3}
# train_df['HomePlanet'] = train_df['HomePlanet'].apply(lambda x : HomePlanet_dict[x])

# Destination_dict = {'55 Cancri e':1,'PSO J318.5-22':2,'TRAPPIST-1e':3}
# train_df['Destination'] = train_df['Destination'].apply(lambda x : Destination_dict[x])

# cabin_3_dict = {'P':1,'S':2}
# train_df['cabin_3'] = train_df['cabin_3'].apply(lambda x : cabin_3_dict[x])

# cabin_1_dict = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8}
# train_df['cabin_1'] = train_df['cabin_1'].apply(lambda x : cabin_1_dict[x])

# train_df['PassengerId_1'] = train_df['PassengerId'].apply(lambda x : x.split('_')[0])
# family_count = train_df.groupby(['PassengerId_1']).size().reset_index(name = 'fcount')
# family_count['fcount'] = family_count['fcount'] > 1
# family_count['fcount'] = family_count['fcount'].astype(int)

# train_df = pd.merge(train_df,family_count,on = 'PassengerId_1')

train_df = train_df.drop(['PassengerId','Name'],axis = 1)

train_df_reserve = train_df.copy()

#--------- model building ---------


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from catboost import CatBoostClassifier

train_df = train_df_reserve

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
numerical_df = train_df[numerical_columns]

# Fit and transform the numerical features
scaled_numerical = scaler.fit_transform(numerical_df)

# Replace the original numerical features with the scaled ones in the original DataFrame
train_df[numerical_columns] = scaled_numerical

# Assuming 'Transported' is your target variable, and 'train_df' is your DataFrame
# You need to handle categorical variables (object type) by encoding them

# One-Hot Encoding for categorical variables
train_df = pd.get_dummies(train_df, columns=['HomePlanet', 'Destination', 'cabin_3','cabin_1'])
train_df.dtypes
# Splitting the data into features (X) and target variable (y)
X = train_df.drop('Transported', axis=1)
y = train_df['Transported']




# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building a Random Forest Classifier (you can choose a different algorithm based on your problem)
model = RandomForestClassifier(random_state=42,n_estimators=200,max_depth=7 )
model = AdaBoostClassifier(random_state=1,n_estimators=750,learning_rate=0.7)
model= GradientBoostingClassifier(learning_rate=0.008,random_state=1,n_estimators=5000)
model = xgb.XGBClassifier(random_state=1,learning_rate=0.05,n_estimators = 500)

# model = XGBClassifier(objective='binary:logistic', random_state=42,max_depth= 3,n_estimator=200)
# model = LinearDiscriminantAnalysis()

model.fit(X_train, y_train)


# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print("Classification Report:\n", report)

error_analysis = X_test.copy()
error_analysis['y_test'] = y_test
error_analysis['y_pred'] = y_pred
error_analysis.to_csv('./Documents/Vamsi/DS/Kaggle/Spaceship Titanic/error_analysis.csv',index = False)

# reading test df

test_df = pd.read_csv('./Documents/Vamsi/DS/Kaggle/Spaceship Titanic/test.csv')

test_passenger_ids = test_df[['PassengerId']]

# test df transformations

test_df['cabin_1'] = test_df['Cabin'].apply(lambda x : x.split('/')[0] if not pd.isna(x) else x )
test_df['cabin_3'] = test_df['Cabin'].apply(lambda x : x if pd.isna(x) else x.split('/')[2] if len(x.split('/')) == 3 else x )

test_df = test_df.drop('Cabin',axis = 1)


test_df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']] = test_df[['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']].fillna(0)

test_df['total_spent'] = test_df['RoomService'] + test_df['FoodCourt'] + test_df['ShoppingMall'] + test_df['Spa'] + test_df['VRDeck']

test_df['total_spent'] = test_df['total_spent'] > 0

categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'cabin_1', 'cabin_3','VIP','total_spent']

# Filling missing values with the most frequent value (mode)

for column in categorical_columns:
    mode_value = test_df[column].mode().iloc[0]  # Get the most frequent value
    test_df[column].fillna(mode_value, inplace=True)

continuous_columns = ['Age']

# Filling missing values with the mean

for column in continuous_columns:
    mean_value = test_df[column].mean()
    test_df[column].fillna(mean_value, inplace=True)

# test_df['Age'] = test_df['Age'].apply(lambda x : math.floor(x/10))
    

test_df.isna().sum()

test_df['VIP'] = test_df['VIP'].astype(int)
test_df['total_spent'] = test_df['total_spent'].astype(int)
test_df['CryoSleep'] = test_df['CryoSleep'].astype(int)

test_df = test_df.drop(['PassengerId','Name'],axis = 1)


# applying model on test data

test_df = pd.get_dummies(test_df, columns=['HomePlanet', 'Destination', 'cabin_1', 'cabin_3'])

test_df = test_df[list(X_test.columns)]
y_pred_test = model.predict(test_df)

test_passenger_ids['Transported'] = y_pred_test
test_passenger_ids['Transported'] = test_passenger_ids['Transported'].astype(bool)

test_passenger_ids.to_csv('./Documents/Vamsi/DS/Kaggle/Spaceship Titanic/submission_10.csv',index = False)


