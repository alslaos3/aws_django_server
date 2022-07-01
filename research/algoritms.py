import json
import numpy as np
import pandas as pd

# load dataset

df = pd.read_csv('https://raw.githubusercontent.com/alslaos3/dataset/main/LungCancer_TRAIN.csv')
# fill missing values
df = df.fillna(0)
x_cols = [c for c in df.columns if c != 'Target']

# set input matrix and target column
X = df[x_cols]
y = df['Target']

# data split train / test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 999)

# train Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 200)
rf = rf.fit(X_train, y_train)

# save RF algorithm
import joblib
joblib.dump(rf, "./random_forest.joblib", compress=True)

