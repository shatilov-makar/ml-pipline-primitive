import pickle
from sklearn.linear_model import LinearRegression
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
model_name = 'model.sav'
path = 'test'
test_data = []
for path, dirs, files in os.walk(path):
    for filename in files:      
        test_data.append(pd.read_csv('{0}/{1}'.format(path,filename),index_col=0))
test_data = pd.concat(test_data, axis = 0,ignore_index=True)
X = np.array(test_data.Time).reshape(-1, 1)
y = np.array(test_data.Temperature).reshape(-1, 1)
x_pol = to_polynom(X, order = 3)
loaded_model = pickle.load(open(model_name, 'rb'))
result = loaded_model.score(x_pol, y)
print(f"R2-score: {result}")
