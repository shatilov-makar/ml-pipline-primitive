import subprocess
import sys

subprocess.run(["pip", "install", '-r','requirements.txt'])

from sklearn import datasets
import numpy as np
import pandas as pd
import os


print("Data_creation started")
if (not os.path.exists("train")):
    print("Folder 'train' was created")
    os.mkdir("train")

if (not os.path.exists("test")):
    print("Folder 'test' was created")
    os.mkdir("test")

print("Data is generating...")
# Пускай будет три тренировочных набора, один тестовый
diabetes_X, diabetes_y = datasets.load_diabetes(
    return_X_y=True, as_frame=True, scaled=False)
diabetes_X['target'] = diabetes_y
ds = np.array_split(diabetes_X, 4)

for i, dataset in enumerate(ds[:3]):
    pd.DataFrame(dataset).to_csv(f"train/train {i + 1}.csv", sep=',')

ds[3].to_csv(f"test/test 1.csv", sep=',')
print("Data was generated!")
