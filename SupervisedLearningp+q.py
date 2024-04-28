###########################
### MODULES NECESSAIRES ###
###########################


# Import the necessary libraries
import numpy as np
import pandas as pd
import math
import fonctionsSupervisedLearning as fsl

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

from auxFonctions import AminoAcid

df = open('data/df_exploitable.csv', 'r')
df = pd.read_csv(df, sep=',', index_col=0)


def create_model_pq(df, kernel = 'rbf', C = 10, p = 2, q = 13, random_state = 42):
    gamma = 0.1
    
    X_train, X_test, pos_train, pos_test = fsl.test_train_split_random_pos(df, p+q, random_state=random_state)
    pos_train = np.array(pos_train==p)
    print(X_train[pos_train==1])
    pos_test = np.array(pos_test==p)
    svm_model = svm.SVC(kernel=kernel, C=C,gamma=gamma, random_state=random_state)
    svm_model.fit(X_train, pos_train)
    pos_predict = svm_model.predict(X_test)
    accuracy = accuracy_score(pos_test,pos_predict)
    # accuracy = 0
    # svm_model = 0
    return svm_model,accuracy

def find_cleavage_pq (X, svm_model, p:int = 2, q:int = 13, nb_letters = 26):
    '''
    find the position of the cleavage site in the primary structure using two SVM models
    /!\ the models must be trained before using this function with the same n and nb_letters as the ones used in this function
    ### Parameters:
    - X: the primary structure as a vector
    - svm_model_in: the SVM model that predicts if the subsequence contains the cleavage site
    - svm_model_pos: the SVM model that predicts the position of the cleavage site in the subsequence
    - threshold: the threshold for the confidence of the prediction
    ### Returns:
    - the position of the cleavage site if the prediction is confident enough, otherwise Nan
    '''
    positions = []
    for i in range(p*nb_letters, len(X)- q*nb_letters, nb_letters):
        test_sub = X[i-p*nb_letters :i + q*nb_letters]
        
        if svm_model.predict([test_sub]):
            position = i//nb_letters
            # positions.append(position.item())
            return position
    return math.nan

