# Pandas is used for data manipulation
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# Read in data and display first 5 rows
train_features = pd.read_csv('train.csv')

train_labels = train_features.loc[(train_features['Target'].notnull()), ['Target']]
train_features = train_features.loc[(train_features['Target'].notnull()), ['SQBage', 'SQBhogar_total', 'SQBedjefe', 'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']]

train_labels = train_labels.values.ravel()
# train_features = train_features.drop(['Target', 'Id', 'idhogar'], axis=1)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 92)
rf.fit(train_features.head(), train_labels[:5])
print(rf.feature_importances_)