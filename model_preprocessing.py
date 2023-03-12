from sklearn import preprocessing
import itertools
import numpy as np
import pandas as pd
import os

path1 = 'test'
path2 = 'train'
is_first = True
print("Model_preprocessing started")

print("StandardScaler applying...")

for path, dirs, files in itertools.chain(os.walk(path1), os.walk(path2)):
    for filename in files:
        data = pd.read_csv('{0}/{1}'.format(path, filename), index_col=0)
        columns = data.columns[data.columns != 'target']
        data[columns] = preprocessing.StandardScaler(
        ).fit_transform(data[columns])
        data.to_csv('{0}/{1}'.format(path, filename), sep=',')
print("Data was preprocessed using StandardScaler!")
