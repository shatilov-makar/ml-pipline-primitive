from sklearn.linear_model import LinearRegression
import pickle
import numpy as np 
import pandas as pd
import os
def to_polynom(x, order = 1, add_bias = False):
    order_range = range( 0 if add_bias else 1, order+1,1)
    x = np.atleast_1d(x)[:]    
    out = np.array([])
    for i in order_range:
        out = np.append(out, np.power(x,i))
    return out.reshape(-1, x.size).T
path = 'train'
train_data = []
for path, dirs, files in os.walk(path):
    for filename in files:      
        train_data.append(pd.read_csv('{0}/{1}'.format(path,filename),index_col=0))
train_data = pd.concat(train_data, axis = 0,ignore_index=True)
X = np.array(train_data.Time).reshape(-1, 1)
y = np.array(train_data.Temperature).reshape(-1, 1)
x_pol = to_polynom(X, order = 3)
reg = LinearRegression().fit(x_pol, y)
filename = 'model.sav'
pickle.dump(reg, open(filename, 'wb'))
