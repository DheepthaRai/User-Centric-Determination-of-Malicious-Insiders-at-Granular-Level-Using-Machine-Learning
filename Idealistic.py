
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pyautogui
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import recall_score, classification_report, f1_score, accuracy_score, confusion_matrix

val=101

def Idealistic(filename, MLalgo):
    global val
    data = pd.read_csv(filename)
    removed_cols = ['user','day','week','starttime','endtime','sessionid','insider']
    x_cols = [i for i in data.columns if i not in removed_cols]

    run = 1
    np.random.seed(run)

    dataTrain = data.sample(frac=0.50)
    unKnownTestUsers = list(set(data.user) - set(dataTrain.user))

    xTrain = dataTrain[x_cols].values
    yTrain = dataTrain['insider'].values
    yTrainBin = yTrain > 0

    xTest = data[data.user.isin(unKnownTestUsers)][x_cols].values
    yTest = data[data.user.isin(unKnownTestUsers)]['insider'].values
    yTestBin = yTest > 0


    if(MLalgo=='RF'):
        ml = RandomForestClassifier(n_jobs=-1, n_estimators=100, max_features='sqrt', max_depth=10)
        ml.fit(xTrain, yTrainBin)
    elif(MLalgo=='LR'):
        ml = LogisticRegression(n_jobs=-1, max_iter=1000, solver='lbfgs', penalty='l2').fit(xTrain, yTrainBin)
    elif(MLalgo=='NN'):
        ml = MLPClassifier(solver='adam', batch_size=200, learning_rate='adaptive', n_iter_no_change=250, hidden_layer_sizes=(20,10,)).fit(xTrain, yTrainBin)
    else:
        ml = XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=5, colsample_bytree=0.7)
        ml.fit(xTrain, yTrainBin)


    target_names=['Normal','Malicious']
    print(classification_report(yTestBin, ml.predict(xTest), target_names=target_names))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(ml, xTest, yTestBin, ax=ax, alpha=0.8)
    plt.show(block=False)
    plt.pause(3)
    myScreenshot = pyautogui.screenshot()
    myScreenshot.save(r'C:\Users\pooja\OneDrive\Pictures\Screenshots\i\r' + str(val) + '.jpg')
    val = val + 1
    plt.close()

    cm=confusion_matrix(yTestBin, ml.predict(xTest))

    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print("False positive rate[Normal, Malicious]: ",FPR)
    print('\n')

'''
print('Performance Evaluation(Instance based results) of insider threat detection using Random Forest Classifier at different granularity levels in Idealistic Scenario:\n')

print('1.\tDay:')
Idealistic('dayr5.2.csv','RF')

print('2.\tWeek:')
Idealistic('weekr5.2.csv','RF')

print('3.\tSession:')
Idealistic('sessionr5.2.csv','RF')



print('Performance Evaluation(Instance based results) of insider threat detection using Logistic Regression at different granularity levels in Idealistic Scenario:\n')

print('1.\tDay:')
Idealistic('dayr5.2.csv','LR')

print('2.\tWeek:')
Idealistic('weekr5.2.csv','LR')

print('3.\tSession:')
Idealistic('sessionr5.2.csv','LR')




print('Performance Evaluation(Instance based results) of insider threat detection using Neural Network at different granularity levels in Idealistic Scenario:\n')

print('1.\tDay:')
Idealistic('dayr5.2.csv','NN')

print('2.\tWeek:')
Idealistic('weekr5.2.csv','NN')

print('3.\tSession:')
Idealistic('sessionr5.2.csv','NN')



print('Performance Evaluation(Instance based results) of insider threat detection using XGBoost Classifier at different granularity levels in Idealistic Scenario:\n')

print('1.\tDay:')
Idealistic('dayr5.2.csv','XG')

print('2.\tWeek:')
Idealistic('weekr5.2.csv','XG')

print('3.\tSession:')
Idealistic('sessionr5.2.csv','XG')


    dataTrain = data.sample(frac=0.50)
    dataTest=data.sample(frac=0.5)

    xTrain = dataTrain[x_cols].values
    yTrain = dataTrain['insider'].values
    yTrainBin = yTrain > 0

    xTest = dataTest[x_cols].values
    yTest = dataTest['insider'].values
    yTestBin = yTest > 0
'''