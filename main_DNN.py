# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 19:48:36 2020

@author: DanhaeSway
"""
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
from sklearn.model_selection import cross_val_score, KFold, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, scale, StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

## Determined that the model should have simplified hidden layers or else it will overfit
def initializeModel():
    model = keras.Sequential()
    model.add(keras.layers.Dense(8,activation='relu', input_shape=(10,)))
    # model.add(keras.layers.Dense(32, activation='relu'))
    # model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(8, activation='relu'))
    # model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.Dense(3, activation='relu'))
    model.add(keras.layers.Dense(2, activation='softmax',input_shape=(2,)))
    
    return model


def scale_df(data):
    continuous_var = ['Age', 'Fare','Family']
    copy_data = data.dropna(axis=0)
    scaled_data = copy_data.copy(deep=True)
    for i in continuous_var:    
        scaled_data[i] = scale(copy_data[i].to_numpy().reshape(-1,1),axis=0)
        
    return scaled_data

if __name__ == '__main__':
    
    model = initializeModel()    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
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
    ##Do final cleaning on NA and removing the label data
    
    ##Start with StandardScaler on the training dataset. Apply the same scaling factor to the test dataset
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
    acc, val_acc, loss,val_loss = [],[],[],[]
    ## Running k-folds classification to improve generalization and reduce overfitting
    K = StratifiedShuffleSplit(10, train_size=0.6)
    for train_index, test_index in K.split(train_data_dropped, train_labels):
        x_train, y_train = train_data_dropped[train_index], train_labels[train_index]
        x_valid, y_valid = train_data_dropped[test_index],train_labels[test_index]
        # ##Only need to balance the training data, not the validation data.
        # x_train, x_valid, y_train, y_valid = train_test_split(train_data_dropped, 
        #                                                       train_labels, test_size=0.2, 
        #                                                       shuffle= True)
        
        y_train = pd.get_dummies(y_train).to_numpy()
        y_valid = pd.get_dummies(y_valid).to_numpy()
        history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_valid, y_valid), verbose=2,)
        acc.append(history.history['accuracy'])
        val_acc.append(history.history['val_accuracy'])
        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
    
    acc = np.hstack(acc)
    val_acc = np.hstack(val_acc)
    loss = np.hstack(loss)
    val_loss = np.hstack(val_loss)
    t = np.vstack([acc,val_acc]).T
    l = np.vstack([loss, val_loss]).T
    plt.figure()
    plt.subplot(311)
    plt.plot(t)
    plt.legend(['Accuracy','Validation Accuracy'])
    plt.subplot(312)
    plt.plot(l)
    plt.legend(['Loss', 'Validation Loss'])
    
    testset = model.predict(test_data)
    plt.subplot(313)
    plt.plot(testset)
    plt.show()
    
    trainset = model.predict(train_data_dropped)
    train_predicted = np.argmax(trainset,axis=1)
    t = np.vstack([train_labels, train_predicted]).T
    plt.figure()
    plt.plot(t)
    print(sum(train_labels==train_predicted))
    
    ##Then export the results
    submission = pd.DataFrame({
            "PassengerId": PassengerId_test,
            "Survived": np.argmax(testset,axis=1)
        })
    submission.to_csv(DEFAULT_DIR+ 'submission.csv', index=False)
    