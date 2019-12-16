from pandas import read_csv, DataFrame, Series
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import mlab

from sklearn.model_selection import train_test_split


"""Read Csv files"""
dfTrain = read_csv('train.csv')
dfTest = read_csv('test.csv')
dfTrain.info()

dfTrain["CAge"]=pd.cut(dfTrain["Age"], bins = [0,10,18,40,max(dfTrain["Age"])] ,labels=["Child","MYoung","Young","Older"])
dfTest["CAge"]=pd.cut(dfTest["Age"], bins = [0,10,18,40,max(dfTest["Age"])] ,labels=["Child","MYoung","Young","Older"])

"""Make dummy variables for categorical data"""
dfTrain= pd.get_dummies(data = dfTrain, dummy_na=True, prefix= ["Pclass","Sex","Embarked","Age"] ,columns=["Pclass","Sex","Embarked","CAge"])
dfTest= pd.get_dummies(data = dfTest, dummy_na=True, prefix= ["Pclass","Sex","Embarked","Age"] ,columns=["Pclass","Sex","Embarked","CAge"])
dfCategor = ["Pclass", "Sex", "Age", "Embarked"]
"""Store the train outcomes for survived"""
Y_train=dfTrain["Survived"]

"""Store PassengerId"""
submission=pd.DataFrame()
submission["PassengerId"]=dfTest["PassengerId"]

"""Ignore useless data"""
dfTrain=dfTrain[dfTrain.columns.difference(["Age","Survived","PassengerId","Name","Ticket","Cabin"])]
dfTest=dfTest[dfTest.columns.difference(["Age","PassengerId","Name","Ticket","Cabin"])]

"""handling a Nan value"""
dfTest["Fare"].iloc[dfTest[dfTest["Fare"].isnull()].index] = dfTest[dfTest["Pclass_3.0"]==1]["Fare"].median()

"""Fit Model"""
clf = GaussianNB()
#clf.fit(dfTrain,Y_train)

param_gridBN = {}
scoringBN = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}
gsBN = GridSearchCV(GaussianNB(), return_train_score=True, param_grid=param_gridBN, scoring=scoringBN, cv=10, refit='Accuracy')

gsBN.fit(dfTrain, Y_train)

results = gsBN.cv_results_

print('=' * 100)
print("best params: " + str(gsBN.best_estimator_))
#print("best params: " + str(gsBN.best_params_))
print('best score:', gsBN.best_score_)
print('=' * 100)
print(gsBN.refit_time_)

#----------------------------------------------------------

new_d = pd.read_csv('train.csv')

def harmonize_data(titanic):
    #отсутствующим полям возраста присваивается медианное значение
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    titanic["Age"].median()
    #пол преобразуется в числовой формат
    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
    #пустое место отплытия заполняется наиболее популярным S
    titanic["Embarked"] = titanic["Embarked"].fillna("S")
    # место отплытия преобразуется в числовой формат
    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
    # отсутствующим полям суммы отплаты за плавание присваивается медианное значение
    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())
    return titanic
train_harm = harmonize_data(new_d)

data_set = train_harm[['Pclass','Sex','Age','Fare','SibSp','Cabin']]

one_hot_encoded_training_predictors = pd.get_dummies(data_set)
one_hot_encoded_training_predictors.head()
X = one_hot_encoded_training_predictors
y = new_d['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.319, random_state=1)
logreg = LogisticRegression() #logistic regression using python
#logreg.fit(X_train, y_train)

#y_pred = logreg.predict(X_test) #predicting the values
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

param_grid = {'C': np.arange(1e-05, 3, 0.1)}
scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}
gs = GridSearchCV(LogisticRegression(max_iter=1000), return_train_score=True,
                  param_grid=param_grid, scoring=scoring, cv=10, refit='Accuracy')

gs.fit(X, y)
results = gs.cv_results_

print('=' * 100)
print("best params: " + str(gs.best_estimator_))
print("best params: " + str(gs.best_params_))
print('best score:', gs.best_score_)
print('=' * 100)

print(gs.refit_time_)

s = [gs.best_score_, gsBN.best_score_]
x = range(len(s))
ax = plt.gca()
ax.bar(x, s, align='center')
ax.set_xticks(x)
ax.set_xticklabels(('Logistic Regression', 'Naive Bayes'))
plt.show()

