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
import matplotlib.pyplot as plt
import matplotlib


train=pd.read_excel('C:\\Users\\juhi\\Documents\\python\\kaggle\\TITANIC\\train.xlsx')
test=pd.read_excel('C:\\Users\\juhi\\Documents\\python\\kaggle\\TITANIC\\test.xlsx')




target = train["Survived"].values
full = pd.concat([train, test])
#print(full.head())
#print(full.describe())
#print(full.info())
def num_missing(x):
  return sum(x.isnull())
  
print (full.apply(num_missing, axis=0)) 


full['surname'] = full["Name"].apply(lambda x: x.split(',')[0].lower())

full["Title"] = full["Name"].apply(lambda x: re.search(' ([A-Za-z]+)\.',x).group(1))
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 2, "Mme": 3,"Don": 9,"Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
full["TitleCat"] = full.loc[:,'Title'].map(title_mapping)




# drop unnecessary columns, these columns won't be useful in analysis and prediction
train = train.drop(['PassengerId','wosname','comb','fin','Name','Ticket'], axis=1)
test    = test.drop(['wosname','comb','fin','Name','Ticket'], axis=1)
train["Embarked"] = train["Embarked"].fillna("C")
train.Embarked=train.Embarked.map({'Q':1, 'S':0, 'C':2})
test.Embarked=test.Embarked.map({'Q':1, 'S':0, 'C':2})

'''
train['Name'] = train.Name.astype("category").cat.codes
test['Name'] = test.Name.astype("category").cat.codes
train['Ticket'] = train.Name.astype("category").cat.codes
test['Ticket'] = test.Name.astype("category").cat.codes
'''
'''
embark_dummies_titanic  = pd.get_dummies(train['Embarked'])

embark_dummies_test  = pd.get_dummies(test['Embarked'])


train = train.join(embark_dummies_titanic)
test    = test.join(embark_dummies_test)

train.drop(['Embarked'], axis=1,inplace=True)
test.drop(['Embarked'], axis=1,inplace=True)
'''
test["RFare"].fillna(test["RFare"].median(), inplace=True)



train['RFare'] = train['RFare'].astype(int)
test['RFare']    = test['RFare'].astype(int)

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
'''
person_dummies_titanic  = pd.get_dummies(train['Sex'])
person_dummies_test  = pd.get_dummies(test['Sex'])
train = train.join(person_dummies_titanic)
test    = test.join(person_dummies_test)

train.drop(['Sex'],axis=1,inplace=True)
test.drop(['Sex'],axis=1,inplace=True)
'''
train.Sex=train.Sex.map({'male':1, 'female':0})
test.Sex=test.Sex.map({'male':1, 'female':0})
corr=train.corr()#["Survived"]
train.corr()['Survived']

train
X_train1 = train.drop("Survived",axis=1)
Y_train1 = train["Survived"]
X_test  = test.drop("PassengerId",axis=1).copy()

#X_train, X_test, Y_train, y_test = train_test_split(X_train1, Y_train1, test_size=0.33, random_state=1)


random_forest = RandomForestClassifier(n_estimators=80,oob_score=True)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

score=random_forest.score(X_train, Y_train)
print ("Score: ", round(score*100, 3))
'''
score1=random_forest.score(X_test, y_test)
print ("Scoretest: ", round(score1*100, 3))


'''
score = random_forest.oob_score_
print ("OOB Score: ", round(score*100, 3))

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





