# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 13:34:48 2020

@author: DanhaeSway
"""


import pandas as pd
import numpy as np
from sklearn import tree

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


DEFAULT_DIR = 'C:/Users/DanhaeSway/Documents/Kaggle/Kaggle_titanic/data/'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
test_data = pd.read_csv(DEFAULT_DIR+TEST_FILE,  delimiter=',')
train_data = pd.read_csv(DEFAULT_DIR+TRAIN_FILE, delimiter=',')


PassengerId_test = test_data['PassengerId']
dropLabels = ['Ticket', 'Cabin', 'Name','PassengerId']
train_data.drop(dropLabels, axis=1, inplace=True)
test_data.drop(dropLabels,axis=1,inplace=True)
train_data['Sex'] = [0 if(i=="male") else 1 for i in train_data['Sex']]
train_data['Embarked'] = train_data['Embarked'].fillna('M')
train_data['Embarked'] = train_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, 'M':3} ).astype(int)
test_data['Sex'] = [0 if(i=="male") else 1 for i in test_data['Sex']]
test_data['Embarked'] = test_data['Embarked'].fillna('M')
test_data['Embarked'] = test_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, 'M':3} ).astype(int)
train_data['Family'] = train_data['SibSp'] + train_data['Parch']
test_data['Family'] = test_data['SibSp'] + test_data['Parch']
del test_data['SibSp'], test_data['Parch'], train_data['SibSp'], train_data['Parch']
##Perform final cleaning on NA and removing the label data

##Start with StandardScaler on the training dataset 
scaler = StandardScaler()
df = train_data.fillna(np.mean(train_data['Age']))
scaled_data = scaler.fit_transform(df[['Age', 'Fare']])
df[['Age','Fare']] = scaled_data
categorical = ['Embarked', 'Pclass']
for var in categorical:
    df = pd.concat([df,  pd.get_dummies(df[var], prefix=var)], axis=1)
    del df[var]
df.drop('Embarked_3', axis=1,inplace=True)
traindf = df
train_data_dropped = df.to_numpy()

df = test_data.fillna(np.mean(train_data['Age']))
scaled_data = scaler.transform(df[['Age', 'Fare']])
df[['Age','Fare']] = scaled_data
for var in categorical:
    df = pd.concat([df,  pd.get_dummies(df[var], prefix=var)], axis=1)
    del df[var]
testdf = df
test_data = df.to_numpy()
train_labels = train_data_dropped[:,0]
train_data_dropped = train_data_dropped[:,1:]
    
### Running classification
clf = tree.DecisionTreeClassifier(max_depth=7)
clf = clf.fit(train_data_dropped, train_labels)
# tree.plot_tree(clf)
acc = clf.score(train_data_dropped, train_labels)
print(acc)

## Running predictions of simple decision tree on test dataset
probDT = clf.predict_proba(test_data)
y_predDT = clf.predict(test_data).astype(int)
acc = clf.score(test_data, y_predDT)

## Visualization commented out 
# tree.export_text(clf, feature_names=train_data.columns)
# print(r)

# plt.figure(figsize=(12,12))
# plt.title('Pearson Correlation of Features', y=1.05, size=15)
# plt.imshow(train_data.astype(float).corr())
# plt.show()

## Random forest?
### Running classification
acc, val_acc, loss,val_loss, store_prob = [],[],[],[],[]

## Running k-folds classification to improve generalization and reduce overfitting. Attempting to 
## Average across tests. Possibly not a valid technique, but I was curious about the outcome.
## 1. Run 200 fold with maximum tree depth of 3 to prevent overfitting
## 2. Then, run 100 fold with maximum tree depth of 6 to slightly overfit onto edge cases
## 3. Average the accuracy for each sample across the 300 tests, then take argmax
## Model performed slightly better with this method - rose from 77% to 79%
N = 200
K = StratifiedShuffleSplit(N, train_size=0.05)

clf = RandomForestClassifier(max_depth=3, n_estimators=150, 
                             min_samples_split=2, random_state=0, warm_start=False)
for train_index, test_index in K.split(train_data_dropped, train_labels):
        x_train, y_train = train_data_dropped[train_index], train_labels[train_index]
        x_valid, y_valid = train_data_dropped[test_index],train_labels[test_index]
        # ##Only need to balance the training data, not the validation data.
        # x_train, x_valid, y_train, y_valid = train_test_split(train_data_dropped, 
        #                                                       train_labels, test_size=0.2, 
        #                                                       shuffle= True)
        
        # y_train = pd.get_dummies(y_train).to_numpy()
        # y_valid = pd.get_dummies(y_valid).to_numpy()
        history = clf.fit(x_train, y_train)
        acc.append(history.score(train_data_dropped,train_labels))
        probRF = clf.predict_proba(test_data)
        store_prob.append(probRF)
        
        
N = 100
K = StratifiedShuffleSplit(N, train_size=0.05)

clf = RandomForestClassifier(max_depth=6, n_estimators=150, 
                             min_samples_split=2, random_state=0, warm_start=False)
for train_index, test_index in K.split(train_data_dropped, train_labels):
        x_train, y_train = train_data_dropped[train_index], train_labels[train_index]
        x_valid, y_valid = train_data_dropped[test_index],train_labels[test_index]
        # ##Only need to balance the training data, not the validation data.
        # x_train, x_valid, y_train, y_valid = train_test_split(train_data_dropped, 
        #                                                       train_labels, test_size=0.2, 
        #                                                       shuffle= True)
        
        # y_train = pd.get_dummies(y_train).to_numpy()
        # y_valid = pd.get_dummies(y_valid).to_numpy()
        history = clf.fit(x_train, y_train)
        acc.append(history.score(train_data_dropped,train_labels))
        probRF = clf.predict_proba(test_data)
        store_prob.append(probRF)
        
store_prob = np.hstack(store_prob)
y_pred = np.asarray([np.sum([store_prob[:,i] for i in range(0,N*6,2)],axis=0), np.sum([store_prob[:,i] for i in range(1,N*6,2)],axis=0)]).T
y_pred = np.argmax(y_pred,axis=1)
scores = cross_val_score(clf, train_data_dropped, train_labels, cv=10)
clf = clf.fit(train_data_dropped, train_labels)
probRF = clf.predict_proba(test_data)
print(acc)
y_predRF = clf.predict(test_data).astype(int)


##Then export the results
submission = pd.DataFrame({
        "PassengerId": PassengerId_test,
        "Survived": y_pred
    })
submission.to_csv(DEFAULT_DIR+ 'submission.csv', index=False)

##How can we plot the classifications?

