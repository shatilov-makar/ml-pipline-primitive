import numpy as np
import pandas as pd
import os


def true_fun(x, a=np.pi, b=0, f=np.sin):
    x = np.atleast_1d(x)[:]
    a = np.atleast_1d(a)

    f = lambda x: x
    x = np.sum([ai * np.power(x, i + 1) for i, ai in enumerate(a)], axis=0)

    return f(x + b)


def noises(shape, noise_power):
    return np.random.randn(*shape) * noise_power


def dataset(a, b, N=250, x_max=1, noise_power=0, seed=42):
    np.random.seed(seed)
    x = np.linspace(0, x_max, N)
    y_true = np.array([])
    f = None
    for f_ in np.append([], f):
        y_true = np.append(y_true, true_fun(x, a, b, f_))
    y_true = y_true.reshape(-1, N).T
    y = y_true + noises(y_true.shape, noise_power)
    y = y * 3 + 10
    x = x * 10
    return {'Time': list([t[0] for t in np.atleast_2d(x).T]), 'Temperature': list([t[0] for t in y])}


if(not os.path.exists("train")):
   os.mkdir("train")


if(not os.path.exists("test")):
   os.mkdir("test")

train1 = dataset(a = [3,-1,-2], b = 1, N = 360, x_max =1.2, noise_power = 0.05)
pd.DataFrame(data=train1, columns=['Time', 'Temperature']).to_csv("train/train 1.csv", sep=',')

train2 = dataset(a = [3,-1,-2], b = 1, N = 360, x_max =1.2, noise_power = 0.15,seed = 45)
pd.DataFrame(train2, columns=['Time', 'Temperature']).to_csv("train/train 2.csv", sep=',')

train3 = dataset(a = [3,-1,-2], b = 1, N = 360, x_max =1.2,seed = 43)
pd.DataFrame(train3, columns=['Time', 'Temperature']).to_csv("train/train 3.csv", sep=',')

test = dataset(a = [3,-1,-2], b = 1, N = 360, x_max =1.2, noise_power = 0.05, seed = 41)
pd.DataFrame(test, columns=['Time', 'Temperature']).to_csv("test/test 1.csv", sep=',')