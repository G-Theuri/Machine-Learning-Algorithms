import pandas as pd
import numpy as np


#import data visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt


#import required machine learning libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression



#load the data into dataframe
train = pd.read_csv('C:/Users/ereez/Downloads/train.csv')
test = pd.read_csv('C:/Users/ereez/Downloads/test.csv')



#cleaning the data 

#train data
train.drop('Cabin', axis=1, inplace=True) #The column Cabin has too many null values
train['Embarked'].ffill(inplace=True)
train['Age'].fillna(train['Age'].mean(), inplace=True)

#test data
test.drop('Cabin', axis=1, inplace=True)
test['Age'].fillna(test['Age'].mean())



#Data preprocessing

le = LabelEncoder()
cat_train = train.columns[train.dtypes.eq('object')]
cat_test =  test.columns[test.dtypes.eq('object')]

for column in cat_train:
    train[column] = le.fit_transform(train[column])
for column in cat_test:
    test[column] = le.fit_transform(test[column])



#Train the model using Logistic Regression model
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

lr = LogisticRegression(max_iter = 10000)
lr.fit(X_train, y_train)

lr.score(X_test, y_test)