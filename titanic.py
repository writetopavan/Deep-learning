# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 11:27:48 2017

@author: PKS
"""
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC


train=pd.read_excel('C:\\Users\\juhi\\Documents\\python\\kaggle\\TITANIC\\trainf.xlsx')
test=pd.read_excel('C:\\Users\\juhi\\Documents\\python\\kaggle\\TITANIC\\test.xlsx')

print(train)
print(test)

# drop unnecessary columns, these columns won't be useful in analysis and prediction
train = train.drop(['PassengerId','Name','Ticket'], axis=1)
test    = test.drop(['Name','Ticket'], axis=1)
train["Embarked"] = train["Embarked"].fillna("S")

embark_dummies_titanic  = pd.get_dummies(train['Embarked'])

embark_dummies_test  = pd.get_dummies(test['Embarked'])


train = train.join(embark_dummies_titanic)
test    = test.join(embark_dummies_test)

train.drop(['Embarked'], axis=1,inplace=True)
test.drop(['Embarked'], axis=1,inplace=True)

test["Fare"].fillna(test["Fare"].median(), inplace=True)



train['Fare'] = train['Fare'].astype(int)
test['Fare']    = test['Fare'].astype(int)

# get average, std, and number of NaN values in titanic_df
average_age_titanic   = train["Age"].mean()
std_age_titanic       = train["Age"].std()
count_nan_age_titanic = train["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df
average_age_test   = test["Age"].mean()
std_age_test       = test["Age"].std()
count_nan_age_test = test["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

train["Age"][np.isnan(train["Age"])] = rand_1
test["Age"][np.isnan(test["Age"])] = rand_2

train['Age'] = train['Age'].astype(int)
test['Age']    = test['Age'].astype(int)
 
train.drop("Cabin",axis=1,inplace=True)
test.drop("Cabin",axis=1,inplace=True)

person_dummies_titanic  = pd.get_dummies(train['Sex'])
person_dummies_test  = pd.get_dummies(test['Sex'])
train = train.join(person_dummies_titanic)
test    = test.join(person_dummies_test)

train.drop(['Sex'],axis=1,inplace=True)
test.drop(['Sex'],axis=1,inplace=True)

X_train = train.drop("Survived",axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId",axis=1).copy()


random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('subtitanic.csv', index=False)
'''
# Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svc.score(X_train, Y_train)

'''





