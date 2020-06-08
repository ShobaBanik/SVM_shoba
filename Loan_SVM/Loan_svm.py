# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:03:43 2020

@author: Shoba Banik
"""

import pandas as pd
import numpy as np                     
import seaborn as sns                 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv('C:/Users/Shoba Banik/Downloads/Loan_SVM/train.csv')
test = pd.read_csv('C:/Users/Shoba Banik/Downloads/Loan_SVM/test.csv')

train.columns
train.shape


train['Loan_Status'].value_counts()

sns.set_style = ("whitegrid");
sns.FacetGrid(train,hue="Loan_Status",size=4)\
    .map(plt.scatter,"Credit_History","ApplicantIncome")\
    .add_legend();
plt.show()

train['LoanAmount'].median()
train['Loan_Amount_Term'].value_counts()

train.apply(lambda x: len(x.unique()))


train['Dependents'].replace('3+', 3,inplace=True)
test['Dependents'].replace('3+', 3,inplace=True)

train.isnull().sum()

train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train ['Married'].fillna(train['Married'].mode()[0],inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

train.isnull().sum()

test.isnull().sum()

test['Gender'].fillna(test['Gender'].mode()[0], inplace=True)
test ['Married'].fillna(test['Married'].mode()[0],inplace=True)
test['Dependents'].fillna(test['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(test['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(test['Credit_History'].mode()[0], inplace=True)

test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(test['LoanAmount'].median(), inplace=True)

test.isnull().sum()

train['LoanAmount'] = np.log(train['LoanAmount'])
test['LoanAmount'] = np.log(test['LoanAmount'])

X = train.iloc[:, 1:-1].values
y = train.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()

X[:,0] = labelencoder_X.fit_transform(X[:,0])

X[:,1] = labelencoder_X.fit_transform(X[:,1])

X[:,3] = labelencoder_X.fit_transform(X[:,3])

X[:,4] = labelencoder_X.fit_transform(X[:,4])

X[:,-1] = labelencoder_X.fit_transform(X[:,-1])

onehotencoder = OneHotEncoder(categorical_features = [0])

X = onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

accuracies.mean()

from sklearn.metrics import classification_report
classification_report(y_test, y_pred)
