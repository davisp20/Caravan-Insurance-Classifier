# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 11:59:41 2020

@author: pdavi
"""
# import libraries
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE                   # For Oversampling
#from outliers import smirnov_grubbs as grubbs
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import *
from sklearn.ensemble import RandomForestClassifier



import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('caravanData.csv')

var=16 

print(dataset.describe())
print('Variables selected :  ', list(dataset.columns.values[[3,10,16,25,29,31,33,40,41,42,43,44,47,59,61,68]]))

selected = dataset.columns.values[[3,10,16,25,29,31,33,40,41,42,43,44,47,59,61,68]]

X = (dataset[dataset.columns[[3,10,16,25,29,31,33,40,41,42,43,44,47,59,61,68]]].values)

# Normalization - Using MinMax Scaler
min_max_scaler = preprocessing.MinMaxScaler()
#X = min_max_scaler.fit_transform(X)

y = np.vstack(dataset['Target'].values)

print('\n')
print('X and y Input Data:   ', X.shape, y.shape)


X_train_original, X_test2, y_train_original, y_test2 = train_test_split(X, y, test_size=0.3,
                                                                        random_state=42)

print('Training Set Shape:   ', X_train_original.shape, y_train_original.shape)

X_val, X_test, y_val, y_test = train_test_split(X_test2, y_test2, test_size=0.33,random_state=42)
# Used Seed in Partitioning so that Test Set remains same for every Run

print('Validation Set Shape: ', X_val.shape,y_val.shape)
print('Test Set Shape:       ', X_test.shape, y_test.shape)

doOversampling = True

if doOversampling:
# Apply regular SMOTE
    sm = SMOTE()
    X_train, y_train = sm.fit_sample(X_train_original, y_train_original)
    print('Training Set Shape after oversampling:   ', X_train.shape, y_train.shape)
    print(pd.crosstab(y_train,y_train))
else:
    X_train = X_train_original
    y_train = y_train_original

#Decision Tree Classifier
clf_DT = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=10, 
                                min_samples_split=2, min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, max_features=None, 
                                max_leaf_nodes=None, min_impurity_decrease=1e-07)
clf_DT.fit(X_train, y_train)
y_pred_DT = clf_DT.predict(X_val)

#Naive Bayes Classifier
clf_NB = BernoulliNB()
clf_NB.fit(X_train, y_train)
y_pred_NB = clf_NB.predict(X_val)

#NN Classifier
MLPClassifier(activation='relu', alpha=1e-05,
       batch_size='auto', beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(64), learning_rate='constant',
       learning_rate_init=0.001, max_iter=2000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
       tol=0.001, validation_fraction=0.1, verbose=True,
       warm_start=False)
clf_MLP = MLPClassifier(alpha=1e-05, hidden_layer_sizes=(64))

clf_MLP.fit(X_train, y_train)
y_pred_MLP = clf_MLP.predict(X_val)

#Logistic Regression
clf_Log = LogisticRegression(solver='liblinear', max_iter=1000, 
                             random_state=42,verbose=2,class_weight='balanced')

clf_Log.fit(X_train, y_train)
y_pred_Log = clf_Log.predict(X_val)
print("Log Coef: " + str(clf_Log.coef_))
print("Log Intercept: " + str(clf_Log.intercept_))

#Random Forest 
clf_RF = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=15,
                                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                max_features='auto', max_leaf_nodes=None, min_impurity_decrease=1e-07, 
                                bootstrap=True, oob_score=False, n_jobs=1, 
                                random_state=42, verbose=1, warm_start=False, class_weight=None)
clf_RF.fit(X_train, y_train)
y_pred_RF = clf_RF.predict(X_val)

print('       Accuracy of Models       ')
print('--------------------------------')
print('Decision Tree           '+"{:.2f}".format(accuracy_score(y_val, y_pred_DT)*100)+'%')
print('Naive Bayes             '+"{:.2f}".format(accuracy_score(y_val, y_pred_NB)*100)+'%')
print('Neural Network          '+"{:.2f}".format(accuracy_score(y_val, y_pred_MLP)*100)+'%')
print('Logistic Regression     '+"{:.2f}".format(accuracy_score(y_val, y_pred_Log)*100)+'%')
print('Random Forest           '+"{:.2f}".format(accuracy_score(y_val, y_pred_RF)*100)+'%')

#Confusion Matrices
print('Decision Tree  ')
cm_DT = confusion_matrix(y_val,y_pred_DT)
print(cm_DT)
print('\n')

print('Naive Bayes  ')
cm_NB = confusion_matrix(y_val,y_pred_NB)
print(cm_NB)
print('\n')

print('Neural Network  ')
cm_MLP = confusion_matrix(y_val,y_pred_MLP)
print(cm_MLP)
print('\n')

print('Logistic Regression  ')
cm_Log = confusion_matrix(y_val,y_pred_Log)
print(cm_Log)
print('\n')

print('Random Forest  ')
cm_RF = confusion_matrix(y_val,y_pred_RF)
print(cm_RF)
print('\n')

y_test = y_test.reshape(-1)
y_train_original = y_train_original.reshape(-1)

y_pred_train_DT = clf_DT.predict(X_train_original)
y_pred_train_NB = clf_NB.predict(X_train_original)
y_pred_train_MLP = clf_MLP.predict(X_train_original)
y_pred_train_Log = clf_Log.predict(X_train_original)
y_pred_test_DT = clf_DT.predict(X_test)
y_pred_test_NB = clf_NB.predict(X_test)
y_pred_test_MLP = clf_MLP.predict(X_test)
y_pred_test_Log = clf_Log.predict(X_test)
cm_DT_train = confusion_matrix(y_train_original,y_pred_train_DT)
cm_NB_train = confusion_matrix(y_train_original,y_pred_train_NB)
cm_MLP_train = confusion_matrix(y_train_original,y_pred_train_MLP)
cm_Log_train = confusion_matrix(y_train_original,y_pred_train_Log)
cm_DT_test = confusion_matrix(y_test,y_pred_test_DT)
cm_NB_test = confusion_matrix(y_test,y_pred_test_NB)
cm_MLP_test = confusion_matrix(y_test,y_pred_test_MLP)
cm_Log_test = confusion_matrix(y_test,y_pred_test_Log)

print('Decision Tree Classification Matrix  ')
print('Training')
print(cm_DT_train)
print('Validation')
print(cm_DT)
print('Test')
print(cm_DT_test)
print('\n')

print('Naive Bayes Classification Matrix ')
print('Training')
print(cm_NB_train)
print('Validation')
print(cm_NB)
print('Test')
print(cm_NB_test)
print('\n')

print('Neural Network Classification Matrix ')
print('Training')
print(cm_MLP_train)
print('Validation')
print(cm_MLP)
print('Test')
print(cm_MLP_test)
print("\n")

print('Logistic Regression Classification Matrix ')
print('Training')
print(cm_Log_train)
print('Validation')
print(cm_Log)
print('Test')
print(cm_Log_test)
print('\n')

clf = clf_NB

importances_RF = clf_RF.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_RF.estimators_],
             axis=0)
indices1 = np.argsort(importances_RF[0:var])[::-1]

indices = indices1[0:var]
# Print the feature ranking
print("Feature ranking:")

for f in range(var):
    print("%d. %s (%f)" % (f + 1, (dataset.columns.values[[3,10,16,25,29,31,33,40,41,42,43,44,47,59,61,68]]).reshape(-1)[indices[f]], importances_RF[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(14, 3))
plt.title("Most Important Features - Random Forest",size=20)
plt.bar(range(var), importances_RF[indices],
       color="#aa6d0a", yerr=std[indices], align="center")
plt.yticks(size=14)
plt.xticks(range(var), (dataset.columns.values[[3,10,16,25,29,31,33,40,41,42,43,44,47,59,61,68]]).reshape(-1)[indices],rotation='vertical',size=12,color="#201506")
plt.xlim([-1, var])
plt.show()
