from sklearn import preprocessing
import itertools
import numpy as np
import pandas as pd
import os

path1 = 'test'
path2 = 'train'
is_first = True

for path, dirs, files in itertools.chain(os.walk(path1), os.walk(path2)):
    for filename in files:
        data = pd.read_csv('{0}/{1}'.format(path,filename),index_col=0)
        array = np.array(data.Time).reshape(-1, 1)
        if (is_first):
            scaler = preprocessing.StandardScaler().fit(array)
            is_first = False
        data["Time"]  = scaler.transform(array)
        data.to_csv('{0}/{1}'.format(path,filename), sep=',')
