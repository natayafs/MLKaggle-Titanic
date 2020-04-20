import numpy as np
import pandas as pd

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