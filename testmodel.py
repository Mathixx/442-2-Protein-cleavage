import pandas as pd
import math
from auxFonctions import AminoAcid
import fonctionsSupervisedLearning2 as fsl2
import fonctionsSupervisedLearning1 as fsl1

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
import time
# import thundersvm as tsvm

import numpy as np

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import joblib 
import fonctionskernel as fk


# open model
model = joblib.load("best_svm_model.pkl")

# open data
data = pd.read_csv("data/df.csv")
data = fsl2.convert_df_to_vectors2(data)

X = data['P_Structure_vector'][1]
X = np.array(X)
print(X)
print(len(X))
X = X.reshape(1,-1)
print(X)
pos = data['Cleavage_Site'][1]

# predict
def main():
    print(fsl1.find_cleavage2(X, model))
    print(pos)

def test(x,y):
    return fk.SimilarityBLOSUM(x,y)

x = "ABDEDHETDGEHDGDDD"
y = "DGEHDGEDGSHDGEDDD"

if __name__ == "__main__":
    # main()
    test(x,y)
    # print(x[0])

