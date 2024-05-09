import pandas as pd
import math
from auxFonctions import AminoAcid
import fonctionsSupervisedLearning2 as fsl2
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




gram_matrix_probabilistic = np.load("gram_matrix.npy")

param_grid = {
    'C': [0.1, 1, 10],  # Example C values
    'kernel': [fk.RBF_kernelPAM, fk.RBF_kernelBLOSUM, fk.ProbabilisticKernel],  # Example kernels
    # 'kernel' : [fk.ProbabilisticKernel],
    # 'C':[1],
    # Add other parameters if your kernel supports them
}

def train_and_evaluate(X_train, y_train, X_test, y_test, C, kernel_function):
    """
    Train SVM, evaluate with ROC curve and return the model and its AUC score.
    """
    # model = BaggingClassifier(svm.SVC(C=C, kernel=kernel_function, probability=True, class_weight='balanced'), n_jobs=-1)
    model = svm.SVC(C=C, kernel = kernel_function, probability=True)
    model.fit(X_train, y_train)
    
    probabilities = model.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    
    return model, roc_auc




def run_svm_analysis():
    """
    Run SVM analysis for different configurations, save the best classifier and ROC curves.
    """
    data = pd.read_csv("data/df.csv")
    data = fsl2.convert_df_to_vectors2(data)
    print("done")

    #select 4 first rows of data
    data = data[:100]
    
    p = 13
    q = 2
    n = p + q
    # print(data)


    X_train, X_test, bool_train, bool_test = fk.test_train_split_random_pos_proba(data,n)
       # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10],  # Example C values
        'kernel': [fk.RBF_kernelPAM, fk.RBF_kernelBLOSUM, fk.ProbabilisticKernel],  # Example kernels
    }

    # Setup the SVM classifier with GridSearchCV
    model = svm.SVC(probability=True)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs = -1)

    # Start timing and fit GridSearchCV
    start = time.time()
    grid_search.fit(X_train, bool_train)
    end = time.time()

    # Best model evaluation
    print("Best parameters found:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predict on the test set using the best model
    probabilities = best_model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(bool_test, probabilities)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic {best_model.kernel} {best_model.C}')

    plt.legend(loc="lower right")
    plt.savefig('data/ROC_Curve.png')
    plt.close()

    # Save the best model
    joblib.dump(best_model, 'best_svm_model.pkl')
    print(f"Best Model Saved with AUC: {roc_auc}, Time to Run: {end - start}s")



if __name__ == "__main__":
    run_svm_analysis()





