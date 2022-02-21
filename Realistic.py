#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pyautogui
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import recall_score, classification_report, f1_score, accuracy_score, confusion_matrix


val=1
def Realistic(filename, MLalgo):
    global val
    data = pd.read_csv(filename)
    removed_cols = ['user','day','week','starttime','endtime','sessionid','insider']
    x_cols = [i for i in data.columns if i not in removed_cols]

    run = 1
    np.random.seed(run)

    data1stHalf = data[data.week <= max(data.week)/2]
    dataTest = data[data.week > max(data.week)/2]

    selectedTrainUsers =  set(data1stHalf[data1stHalf.insider > 0]['user'])
    nUsers = np.random.permutation(list(set(data1stHalf.user) - selectedTrainUsers))
    trainUsers = np.concatenate((list(selectedTrainUsers), nUsers[:400-len(selectedTrainUsers)]))

    unKnownTestUsers = list(set(dataTest.user) - selectedTrainUsers)

    xTrain = data1stHalf[data1stHalf.user.isin(trainUsers)][x_cols].values
    scaler = preprocessing.StandardScaler().fit(xTrain)
    xTrain = scaler.transform(xTrain)
    yTrain = data1stHalf[data1stHalf.user.isin(trainUsers)]['insider'].values
    yTrainBin = yTrain > 0

    xTest = dataTest[dataTest.user.isin(unKnownTestUsers)][x_cols].values
    scaler = preprocessing.StandardScaler().fit(xTest)
    xTest = scaler.transform(xTest)
    yTest = dataTest[dataTest.user.isin(unKnownTestUsers)]['insider'].values
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
    myScreenshot.save(r'C:\Users\pooja\OneDrive\Pictures\Screenshots\r' + str(val) + '.jpg')
    val=val+1
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
print('Performance Evaluation(Instance based results) of insider threat detection using Random Forest Classifier at different granularity levels in Realistic Scenario:\n')

print('1.\tDay:')
Realistic('dayr5.2.csv','RF')

print('2.\tWeek:')
Realistic('weekr5.2.csv','RF')

print('3.\tSession:')
Realistic('sessionr5.2.csv','RF')


print('4.\tSubsession(N=25):')
Realistic('sessionnact25r5.2.csv','RF')

print('5.\tSubsession(N=50):')
Realistic('sessionnact50r5.2.csv','RF')

print('6.\tSubsession(T=120):')
Realistic('sessiontime120r5.2.csv','RF')

print('7.\tSubsession(T=240):')
Realistic('sessiontime240r5.2.csv','RF')


print('Performance Evaluation(Instance based results) of insider threat detection using Logistic Regression at different granularity levels in Realistic Scenario:\n')

print('1.\tDay:')
Realistic('dayr5.2.csv','LR')

print('2.\tWeek:')
Realistic('weekr5.2.csv','LR')

print('3.\tSession:')
Realistic('sessionr5.2.csv','LR')

print('4.\tSubsession(N=25):')
Realistic('sessionnact25r5.2.csv','LR')

print('5.\tSubsession(N=50):')
Realistic('sessionnact50r5.2.csv','LR')

print('6.\tSubsession(T=120):')
Realistic('sessiontime120r5.2.csv','LR')

print('7.\tSubsession(T=240):')
Realistic('sessiontime240r5.2.csv','LR')


print('Performance Evaluation(Instance based results) of insider threat detection using Neural Network at different granularity levels in Realistic Scenario:\n')

print('1.\tDay:')
Realistic('dayr5.2.csv','NN')

print('2.\tWeek:')
Realistic('weekr5.2.csv','NN')

print('3.\tSession:')
Realistic('sessionr5.2.csv','NN')


print('4.\tSubsession(N=25):')
Realistic('sessionnact25r5.2.csv','NN')

print('5.\tSubsession(N=50):')
Realistic('sessionnact50r5.2.csv','NN')

print('6.\tSubsession(T=120):')
Realistic('sessiontime120r5.2.csv','NN')

print('7.\tSubsession(T=240):')
Realistic('sessiontime240r5.2.csv','NN')


print('Performance Evaluation(Instance based results) of insider threat detection using XGBoost Classifier at different granularity levels in Realistic Scenario:\n')

print('1.\tDay:')
Realistic('dayr5.2.csv','XG')

print('2.\tWeek:')
Realistic('weekr5.2.csv','XG')

print('3.\tSession:')
Realistic('sessionr5.2.csv','XG')

print('4.\tSubsession(N=25):')
Realistic('sessionnact25r5.2.csv','XG')

print('5.\tSubsession(N=50):')
Realistic('sessionnact50r5.2.csv','XG')


print('6.\tSubsession(T=120):')
Realistic('sessiontime120r5.2.csv','XG')

print('7.\tSubsession(T=240):')
Realistic('sessiontime240r5.2.csv','XG')




'''






