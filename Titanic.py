import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

path_test = './data/test.csv'
path_train = './data/train.csv'

df_test = pd.read_csv(path_test)
df_train = pd.read_csv(path_train)

print("")
print("--------Amount of data--------")
print(df_test.shape)
print(df_train.shape)

print("")
print("--------Tipe of data--------")
print(df_test.info())
print(df_train.info())

print("")
print("--------Missing data--------")
print(pd.isnull(df_test).sum())
print(pd.isnull(df_train).sum())

print("")
print("--------Dataset stadistics--------")
print(df_test.describe())
print(df_train.describe())

df_test['Sex'].replace(['female','male'],[0,1],inplace=True)
df_train['Sex'].replace(['female','male'],[0,1],inplace=True)

df_test['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)
df_train['Embarked'].replace(['Q','S','C'],[0,1,2],inplace=True)

print("")
print("--------Ages--------")
print(df_test['Age'].mean())
print(df_train['Age'].mean())

prom=30
df_test['Age'] = df_test['Age'].replace(np.nan, prom)
df_train['Age'] = df_train['Age'].replace(np.nan, prom)

bins = [0,8,15,18,25,40,60,100]
names = ['1','2','3','4','5','6','7']
df_test['Age'] = pd.cut(df_test['Age'], bins, labels = names)
df_train['Age'] = pd.cut(df_train['Age'], bins, labels = names)

df_test.drop(['Cabin'], axis = 1, inplace=True)
df_train.drop(['Cabin'], axis = 1, inplace=True)

df_test.drop(['Name', 'Ticket'], axis = 1)
df_train.drop(['PassengerId', 'Name', 'Ticket'], axis = 1)

df_test.dropna(axis = 0, how = 'any', inplace=True)
df_train.dropna(axis = 0, how = 'any', inplace=True)

print("")
print("--------New dataset--------")
print(pd.isnull(df_test).sum())
print(pd.isnull(df_train).sum())
print(df_test.shape)
print(df_train.shape)
print(df_test.head())
print(df_train.head())

#Train
X = np.array(df_train.drop(['Survived'], 1))
y = np.array(df_train['Survived'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)

print("")
print('Logistic regression:')
print(logreg.score(X_train, y_train))

svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)

print("")
print('Vectors support:')
print(svc.score(X_train, y_train))

knn =KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)

print("")
print('Neighbors:')
print(knn.score(X_train, y_train))

ids = df_test['PassengerId']

prediction_logreg = logreg.predict(df_test.drop('PassengerId', axis = 1))
out_logreg = pd.DataFrame({'PassengerId': ids, 'Survived': prediction_logreg})

print("")
print('Logistic Regression: ')
print(out_logreg.head())

prediction_svc = svc.predict(df_test.drop('PassengerId', axis = 1))
out_svc = pd.DataFrame({'PassengerId': ids, 'Survived': prediction_svc})

print("")
print('Vectors support:')
print(out_svc.head())

prediction_knn = knn.predict(df_test.drop('PassengerId', axis = 1))
out_knn = pd.DataFrame({'PassengerId': ids, 'Survived': prediction_knn})

print("")
print('Neighbors:')
print(out_knn.head())