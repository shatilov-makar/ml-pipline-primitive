from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import pandas as pd
import os

path = 'train'
train_data = []
print("Model_preparation started")

for path, dirs, files in os.walk(path):
    for filename in files:
        train_data.append(pd.read_csv(
            '{0}/{1}'.format(path, filename), index_col=0))

train_data = pd.concat(train_data, axis=0, ignore_index=True)
columns = train_data.columns

y = train_data.pop('target')
X = np.array(train_data)
y = np.array(y)

regr = LinearRegression()
print("LinearRegression created!")
print("LinearRegression is learning...")
regr.fit(X, y)

filename = 'model.sav'
pickle.dump(regr, open(filename, 'wb'))
print(f"LinearRegression was saved as {filename}")
