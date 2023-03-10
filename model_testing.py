import pickle
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import os

model_name = 'model.sav'
path = 'test'
test_data = []
print("Model_testing started")

for path, dirs, files in os.walk(path):
    for filename in files:
        test_data.append(pd.read_csv(
            '{0}/{1}'.format(path, filename), index_col=0))

test_data = pd.concat(test_data, axis=0, ignore_index=True)
y = test_data.pop('target')
X = np.array(test_data)
y = np.array(y)

loaded_model = pickle.load(open(model_name, 'rb'))
result = loaded_model.score(X, y)
print(f"R2-score: {result}")
