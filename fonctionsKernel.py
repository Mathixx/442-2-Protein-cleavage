import pandas as pd
import math
from auxFonctions import AminoAcid
import fonctionsSupervisedLearning2 as fsl2

import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

from Bio.Align import substitution_matrices





def est_un_caractere(obj):
    return isinstance(obj, str) and len(obj) == 1

def Phi(x : chr, y : chr, i : int) :
    if not(est_un_caractere(x) and est_un_caractere(y)) :
        raise ValueError("x and y must be single characters")
    '''
    Fonction servant de base a la Kernel probabiliste
    ### Parameters:
    - x : un acide aminé (sous forme de string)
    - y : un acide aminé (sous forme de string)
    - i : un entier compris entre -p et q-1 (inclus)
    ### Returns:
    - La valeur de la fonction Phi_i(x,y)
    '''
    if (x == y) :
        return ( amino_acid_s_values.loc[x, i] + math.log(1 + math.exp(amino_acid_s_values.loc[x, i])) )
    else :
        return ( amino_acid_s_values.loc[x, i] + amino_acid_s_values.loc[y, i] )

def LogKernel(x : str, y : str) :
    '''
    Fonction servant de base a la Kernel probabiliste
    ### Parameters:
    - x : une sequence d'acides aminés de taille p+q (sous forme de string)
    - y : une sequence d'acides aminés de taille p+q (sous forme de string)
    ### Returns:
    - La valeur de la fonction LogKernel(x,y)
    '''
    sum = 0
    for i in range(-p, q) :
        sum += Phi(x[p+i], y[p+i], i)
        #print("Sum :"+ str(sum))
    return sum

"""
def ProbalisticKernel(X, Y):
    # Initialize an empty matrix to store the kernel values
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))

    # Calculate the kernel value for each pair of samples
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            x_str = fsl2.vector_to_word(x)
            y_str = fsl2.vector_to_word(y)
            gram_matrix[i, j] = math.exp(LogKernel(x_str, y_str))

    return gram_matrix
"""

def ProbKernel(x, y):
    X_str = fsl2.vector_to_word(x)
    Y_str = fsl2.vector_to_word(y)
    '''
    Fonction servant de base a la Kernel probabiliste
    ### Parameters:
    - x : une sequence d'acides aminés converti au prealable en vecteur de taille (p+q)*26 composé de 0 et 1
    - y : une sequence d'acides aminés converti au prealable en vecteur de taille (p+q)*26 composé de 0 et 1
    ### Returns:
    - La valeur de la fonction Kernel(x,y)
    '''
    return math.exp(LogKernel(X_str, Y_str))

def ProbabilisticKernel(X, Y):
    # Initialize an empty matrix to store the kernel values
    gram_matrix = np.zeros((X.shape[0], Y.shape[0]))

    # Calculate the kernel value for each pair of samples
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            gram_matrix[i, j] = ProbKernel(x, y)

    return gram_matrix