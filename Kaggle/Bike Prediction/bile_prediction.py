#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 19:09:55 2023

@author: vamsik
"""

# datetime
# season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
# holiday - whether the day is considered a holiday
# workingday - whether the day is neither a weekend nor holiday
# weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
# temp - temperature in Celsius
# atemp - "feels like" temperature in Celsius
# humidity - relative humidity
# windspeed - wind speed
# casual - number of non-registered user rentals initiated
# registered - number of registered user rentals initiated
# count - number of total rentals


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization

train_df = pd.read_csv('./Documents/Vamsi/DS/Kaggle/Bike Prediction/train.csv')

train_df.info()


train_df['date'] = train_df['datetime'].apply(lambda x: x.split()[0]) 
    
train_df['year'] = train_df['datetime'].apply(lambda x: x.split()[0].split('-')[0])
train_df['month'] = train_df['datetime'].apply(lambda x: x.split()[0].split('-')[1])
train_df['day'] = train_df['datetime'].apply(lambda x: x.split()[0].split('-')[2])
train_df['hour'] = train_df['datetime'].apply(lambda x: x.split()[1].split(':')[0])
train_df['minute'] = train_df['datetime'].apply(lambda x: x.split()[1].split(':')[1])
train_df['second'] = train_df['datetime'].apply(lambda x: x.split()[1].split(':')[2])

from datetime import datetime
import calendar

train_df['weekday'] = train_df['date'].apply( lambda dateString: calendar.day_name[datetime.strptime(dateString, "%Y-%m-%d").weekday()])

train_df['season'] = train_df['season'].map({1: 'Spring',
                                       2: 'Summer',
                                       3: 'Fall',
                                       4: 'Winter'})

train_df['weather'] = train_df['weather'].map({1: 'Clear',
                                         2: 'Mist, Few clouds',
                                         3: 'Light Snow, Rain, Thunderstorm',
                                         4: 'Heavy Rain, Thunderstorm, Snow, Fog'})


import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

mpl.rc('font', size=15) 
sns.displot(train_df['count']) 
sns.displot(np.log(train_df['count']))


#bar plot

# Step 1 :
    
mpl.rc('font', size=14)                       
mpl.rc('axes', titlesize=15)                  
figure, axes = plt.subplots(nrows=3, ncols=2) 
plt.tight_layout()                            
figure.set_size_inches(10, 9)                 

# Step 2 :

sns.barplot(x='year', y='count', data=train_df, ax=axes[0, 0])
sns.barplot(x='month', y='count', data=train_df, ax=axes[0, 1])
sns.barplot(x='day', y='count', data=train_df, ax=axes[1, 0])
sns.barplot(x='hour', y='count', data=train_df, ax=axes[1, 1])
sns.barplot(x='minute', y='count', data=train_df, ax=axes[2, 0])
sns.barplot(x='second', y='count', data=train_df, ax=axes[2, 1])


# Step 3 :
    
axes[0, 0].set(title='Rental amounts by year')
axes[0, 1].set(title='Rental amounts by month')
axes[1, 0].set(title='Rental amounts by day')
axes[1, 1].set(title='Rental amounts by hour')
axes[2, 0].set(title='Rental amounts by minute')
axes[2, 1].set(title='Rental amounts by second')

axes[1, 0].tick_params(axis='x', labelrotation=90)
axes[1, 1].tick_params(axis='x', labelrotation=90)

#box plot

# Step 1 :
        
mpl.rc('font', size=14)                       
mpl.rc('axes', titlesize=15)                  
figure, axes = plt.subplots(nrows=2, ncols=2) 
plt.tight_layout()                            
figure.set_size_inches(10, 13)    

# Step 2 :

sns.boxplot(x='season', y='count', data=train_df, ax=axes[0, 0])
sns.boxplot(x='weather', y='count', data=train_df, ax=axes[0, 1])
sns.boxplot(x='holiday', y='count', data=train_df, ax=axes[1, 0])
sns.boxplot(x='workingday', y='count', data=train_df, ax=axes[1, 1])


# Step 3 :
    
axes[0, 0].set(title='Rental amounts by Season')
axes[0, 1].set(title='Rental amounts by Weather')
axes[1, 0].set(title='Rental amounts by Holiday')
axes[1, 1].set(title='Rental amounts by Working Day')

axes[0, 1].tick_params(axis='x', labelrotation=30)

# point plot
# A point plot shows the mean and confidence intervals of numerical data based on categorical
# data as dots and lines. It provides the same information as a bar graph, 
# but is better suited for plotting multiple graphs on one screen to compare them to each other.

# Step 1 :

mpl.rc('font', size = 11)
figure, axes = plt. subplots(nrows = 5) # 5행 1열
figure.set_size_inches(12, 18)

# Step 2 :

sns.pointplot(x = 'hour', y = 'count', data = train_df, hue = 'workingday', ax = axes[0])
sns.pointplot(x = 'hour', y = 'count', data = train_df, hue = 'holiday', ax = axes[1])
sns.pointplot(x = 'hour', y = 'count', data = train_df, hue = 'weekday', ax = axes[2])
sns.pointplot(x = 'hour', y = 'count', data = train_df, hue = 'season', ax = axes[3])
sns.pointplot(x = 'hour', y = 'count', data = train_df, hue = 'weather', ax = axes[4])

#scatter plot
#Scatter plots with regression lines are used to identify correlations between numerical data.

# Step 1 :

mpl.rc('font', size = 15)
figure, axes = plt.subplots(nrows = 2, ncols = 2) # 2행 2열
plt.tight_layout()
figure.set_size_inches(7, 6)

# Step 2 :

sns.regplot(x = 'temp', y = 'count', data = train_df, ax = axes[0, 0],
           scatter_kws = {'alpha': 0.3}, line_kws = {'color': 'blue'})
sns.regplot(x = 'atemp', y = 'count', data = train_df, ax = axes[0, 1],
           scatter_kws = {'alpha': 0.3}, line_kws = {'color': 'blue'})
sns.regplot(x = 'windspeed', y = 'count', data = train_df, ax = axes[1, 0],
           scatter_kws = {'alpha': 0.3}, line_kws = {'color': 'blue'})
sns.regplot(x = 'humidity', y = 'count', data = train_df, ax = axes[1, 1],
           scatter_kws = {'alpha': 0.3}, line_kws = {'color': 'blue'})

# heat map > to study correlation

corrMat = train_df[['temp', 'atemp', 'humidity', 'windspeed', 'count']].corr()
fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
sns.heatmap(corrMat, annot = True) 
ax.set(title = "Heatmap of Numerical Data")


# model building
train_df_final = train_df.drop(['datetime','month','casual','registered','day','minute','second','date','windspeed'],axis = 1)
train_df_final['count'] = np.log(train_df_final['count'])
train_df_final = train_df_final[train_df_final.weather != 4]

# Machine Learning Libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_log_error, make_scorer

# Machine Learning Models
from sklearn.linear_model import LinearRegression  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor

# Split data into x and y.
X = train_df_final.drop("count", axis=1)
y = train_df_final["count"]

# Split train data into train and test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Make the RMSLE scorer
rmsle_scorer = make_scorer(rmsle)

models = {
    'Linear Regression': LinearRegression(),
    # 'Decision Tree': DecisionTreeRegressor(),
    # 'Random Forest': RandomForestRegressor()
}

for model_name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, scoring=rmsle_scorer, cv=5)
    
    print(f"Model: {model_name}")
    print(f"Average RMSLE: {np.mean(cv_scores)}\n")
    
    
    
