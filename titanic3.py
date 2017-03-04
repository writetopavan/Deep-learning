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
import seaborn as sns
import re
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

train=pd.read_excel('C:\\Users\\juhi\\Documents\\python\\kaggle\\TITANIC\\train.xlsx')
test=pd.read_excel('C:\\Users\\juhi\\Documents\\python\\kaggle\\TITANIC\\test.xlsx')

target = train["Survived"].values
full = pd.concat([train, test])
#def num_missing(x):
 # return sum(x.isnull())
  
#print (full.apply(num_missing, axis=0)) 
full[full.Embarked.isnull()]
median_fare=full[(full['Pclass'] == 3) & (full['Embarked'] == 'S') & (full['SibSp'] == 0) & (full['Parch'] == 0)  ]['Fare'].median()
full["Fare"] = full["Fare"].fillna(median_fare)

full.isnull().sum()

A=full.groupby(['Ticket']).size()
A=pd.DataFrame(A)
A=A.reset_index()
A.columns = ['Ticket', 'cnt']
full=A.merge(full, left_on='Ticket', right_on='Ticket', how='right')

full.describe()
full["Nfare"]=full["Fare"]/full["cnt"]

full['surname'] = full["Name"].apply(lambda x: x.split(',')[0].lower())

'''
pd.options.display.mpl_style = 'default'
full.hist(bins=10,figsize=(10,7),grid=False)

g = sns.FacetGrid(full, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="purple")

g = sns.FacetGrid(full, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Nfare", "Age",edgecolor="w").add_legend()

sns.factorplot(x = 'Embarked',y="Survived", data = full,color="r")

sns.factorplot(x="Sex", y="Survived", col="Pclass",
                    data=full, saturation=.5,
                    kind="bar", ci=None, aspect=.6)

ax = sns.boxplot(x="Survived", y="Age", 
                data=full)
ax = sns.stripplot(x="Survived", y="Age",
                   data=full, jitter=True,
                   edgecolor="gray")
sns.plt.title("Survival by Age",fontsize=12)

full.Age[full.Pclass == 1].plot(kind='kde')    
full.Age[full.Pclass == 2].plot(kind='kde')
full.Age[full.Pclass == 3].plot(kind='kde')
 # plots an axis lable
plt.xlabel("Age")    
plt.title("Age Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')

corr=full.corr()#["Survived"]
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between features')

g = sns.factorplot(x="Age", y="Embarked",
                    hue="Sex", row="Pclass",
                    data=full[full.Embarked.notnull()],
                    orient="h", size=2, aspect=3.5, 
                   palette={'male':"purple", 'female':"blue"},
                    kind="violin", split=True, cut=0, bw=.2)
                    
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=full)
sns.boxplot(x="Embarked", y="Nfare", hue="Pclass", data=full)
'''

full["Title"] = full["Name"].apply(lambda x: re.search(' ([A-Za-z]+)\.',x).group(1))
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 
2, "Mme": 3,"Don": 9,"Dona": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
full["TitleCat"] = full.loc[:,'Title'].map(title_mapping)

full["Deck"]=full.Cabin.str[0]
full.Deck.fillna('N', inplace=True)
full["fsize"]=full["Parch"]+full["SibSp"]+1
full.loc[full["fsize"] == 1, "fsized"] = 1
full.loc[(full["fsize"] > 1)  &  (full["fsize"] < 5) , "fsized"] = 2
full.loc[full["fsize"] >4, "fsized"] = 3
#F=1,M=2, C=3,S=4,H=5,w=6
full.loc[(full['Parch'] > 2) & (full['Sex'] =="male") , 'role'] = 1
full.loc[(full['Parch'] > 2) & (full['Sex'] =="female") , 'role'] = 2
full.loc[(full['Parch'] > 0) & (full['Title'] =="Mrs") , 'role'] = 2
full.loc[(full['Parch'] == 0) & (full['Title'] =="Mrs") , 'role'] = 6
full.loc[(full['Parch'] > 0) & (full['Title'] =="Miss") , 'role'] = 3
full.loc[(full['Parch'] > 0) & (full['Title'] =="Master") , 'role'] = 3

full.loc[full['SibSp'] > 1, 'role'] = 3
full.loc[full['Age'] < 19, 'role'] = 3

full.loc[(full['Parch'] == 0) & (full['SibSp'] == 0) , 'role'] = 4
full.loc[(full.Parch==0) & (full.SibSp==1) & (full.role.isnull()) & (full['Sex'] =="male") , 'role']=5
full.loc[(full.Parch==0) & (full.SibSp==1) & (full.role.isnull()) & (full['Sex'] =="female") , 'role']=6
full.loc[(full['Parch'] == 2) & (full['SibSp'] == 0) & (full.role.isnull()), 'role']=1
full.loc[(full['Parch'] == 1) & (full['SibSp'] == 0) & (full.role.isnull()), 'role']=1
full.loc[(full['Parch'] == 1) & (full['SibSp'] == 1) & (full.role.isnull()), 'role']=1
full.loc[(full['Parch'] == 2) & (full['SibSp'] == 1) & (full.role.isnull()), 'role']=1

full[full.role.isnull()][['Parch' , 'SibSp']].drop_duplicates()
full[(full['Parch'] == 2) & (full['SibSp'] == 1) & (full.role.isnull())]
full[(full['Age'] < 19)  & (full.role.isnull())]
full[(full['surname'] == "johnston")]
B=full['Age'].groupby(full['role']).median()
B=pd.DataFrame(B)
B=B.reset_index()
B.columns = ['role', 'rage']

B.rage[B.role==1].values[0]
def setage(df):
    if np.isnan(df.Age):
        return B["rage"][B.role==df["role"]].values[0]
    else:
        return df["Age"]
full['Age'] = full.apply(setage, axis=1)
full["Embarked"] = full["Embarked"].fillna("C")
full.head()
full.Embarked=full.Embarked.map({'Q':1, 'S':0, 'C':2})
full['Deck'] = full.Deck.astype("category").cat.codes
full["nlen"] = full["Name"].apply(lambda x: len(x))
bins = [0, 20, 40, 57, 85]
group_names = [1, 2, 3, 4]
full['nlend'] = pd.cut(full['nlen'], bins, labels=group_names)

sns.factorplot(x="TitleCat", y="Survived", data=full)
#full.to_csv('subtitanic1.csv', index=False)
#full = full.drop(['role','cnt'], axis=1)
#full.describe()
sns.factorplot(x="fsized", y="Survived", data=full)
test=full[full.Survived.isnull()]
train=full[full.Survived.notnull()]
# drop unnecessary columns, these columns won't be useful in analysis and prediction
train = train.drop(['PassengerId','Name','Ticket','Cabin','surname','Title','fsize','Parch','SibSp','nlen'], axis=1)
test    = test.drop(['Cabin','Name','Ticket','surname','Title','Survived','fsize','Parch','SibSp','nlen'], axis=1)
#test    = test.drop(['Survived'], axis=1)

std_scale = preprocessing.StandardScaler().fit(X_train[['Nfare','Fare','Age']])
X_train[['Nfare','Fare','Age']]=std_scale.transform(X_train[['Nfare','Fare','Age']])
X_test[['Nfare','Fare','Age']]=std_scale.transform(X_test[['Nfare','Fare','Age']])

train.Sex=train.Sex.map({'male':1, 'female':0})
test.Sex=test.Sex.map({'male':1, 'female':0})
corr=train.corr()#["Survived"]
train.corr()['Survived']


X_train1 = train.drop("Survived",axis=1)
Y_train1 = train["Survived"]
X_test  = test.drop("PassengerId",axis=1).copy()

#X_train, X_test, Y_train, y_test = train_test_split(X_train1, Y_train1, test_size=0.33, random_state=1)

'''
mnb = GaussianNB()

mnb.fit(X_train, Y_train)

Y_pred = mnb.predict(X_test)

scorem=mnb.score(X_train, Y_train)
print ("Score: ", round(scorem*100, 3))

scorem1=mnb.score(X_test, y_test)
print ("Scoretest: ", round(scorem1*100, 3))
'''
logistic = linear_model.LogisticRegression()

logistic.fit(X_train, Y_train)

Y_predl = logistic.predict(X_test)

scorem=logistic.score(X_train, Y_train)
print ("Score: ", round(scorem*100, 3))

scorem1=logistic.score(X_test, y_test)
print ("Scoretest: ", round(scorem1*100, 3))


random_forest = RandomForestClassifier(n_estimators=100,random_state=1, max_depth=9,min_samples_split=6, min_samples_leaf=4)

random_forest.fit(X_train, Y_train)

Y_predr = random_forest.predict(X_test)

score=random_forest.score(X_train1, Y_train1)
print ("Score: ", round(score*100, 3))

score1=random_forest.score(X_test, y_test)
print ("Scoretest: ", round(score1*100, 3))

train.drop(['PassengerId','Name','Ticket','Cabin','surname','Title','fsize','Parch','SibSp'],



neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, Y_train)

Y_predn = neigh.predict(X_test)

score=neigh.score(X_train1, Y_train1)
print ("Score: ", round(score*100, 3))

score1=neigh.score(X_test, y_test)
print ("Scoretest: ", round(score1*100, 3))

from sklearn.ensemble import VotingClassifier
eclf1 = VotingClassifier(estimators=[
        ('lr', logistic), ('rf', random_forest), ('bgh', neigh)], voting='soft')
eclf1 = eclf1.fit(X_train1, Y_train1)
predictions=eclf1.predict(X_test)
predictions
from sklearn.metrics import accuracy_score
accuracy_score(Y_train, predictions)

test_predictions=eclf1.predict(X_test)
accuracy_score(y_test, test_predictions)

test_predictions=test_predictions.astype(int)
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("titanic_submission.csv", index=False)


score = random_forest.oob_score_
print ("OOB Score: ", round(score*100, 3))


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('subtitanic2.csv', index=False)


'''
# Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svc.score(X_train, Y_train)

'''





