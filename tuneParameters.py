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

from sklearn.calibration import CalibratedClassifierCV, calibration_curve




gram_matrix_probabilistic = np.load("gram_matrix.npy")


def train_and_evaluate(X_train, bool_train, X_test, bool_test, C, kernel_function):
    """
    Train SVM, evaluate with ROC curve and return the model and its AUC score.
    """
    # model = BaggingClassifier(svm.SVC(C=C, kernel=kernel_function, probability=True, class_weight='balanced'), n_jobs=-1)
    model = svm.SVC(C=C, kernel = kernel_function, probability=True)
    model.fit(X_train, bool_train)
    
    probabilities = model.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(bool_test, probabilities)
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
        'C': [0.1,0.05, 1],  # Example C values
        # 'kernel': [fk.RBF_kernelPAM, fk.RBF_kernelBLOSUM, fk.ProbabilisticKernel],  # Example kernels
        'kernel' : [fk.RBF_kernelBLOSUM, fk.RBF_kernelPAM, 'rbf', fk.ProbabilisticKernel]
    }

    # Setup the SVM classifier with GridSearchCV
    model = svm.SVC(probability=True)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs = -1, return_train_score=True)

    # Start timing and fit GridSearchCV
    start = time.time()
    grid_search.fit(X_train, bool_train)
    end = time.time()

    # Best model evaluation
    print("Best parameters found:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Predict on the test set using the best model
    probabilities = best_model.predict_proba(X_test)[:,1]
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
    plt.savefig('data/ROC_Curve_test.png')
    plt.close()

    # Save the best model
    joblib.dump(best_model, 'best_svm_model_test.pkl')
    print(f"Best Model Saved with AUC: {roc_auc}, Time to Run: {end - start}s")
    with open("data/ROC_AUC.txt", "w") as f:
        f.write(f"Best Model Saved with AUC: {roc_auc}, Time to Run: {end - start}s")
    # Save GridSearchCV results to a text file
    with open("data/GridSearchCV_results.txt", "w") as f:
        f.write("Best parameters found: {}\n".format(grid_search.best_params_))
        f.write("GridSearchCV results:\n")
        for i, params in enumerate(grid_search.cv_results_['params']):
            f.write("Configuration {}: {}\n".format(i+1, params))
            f.write("Mean train score: {}\n".format(grid_search.cv_results_['mean_train_score'][i]))
            f.write("Mean test score: {}\n".format(grid_search.cv_results_['mean_test_score'][i]))
            f.write("\n")


    ##à supprimer

    raw_scores = best_model.predict_proba(X_test)[:,1]

# Calibration curve
    prob_true, prob_pred = calibration_curve(bool_test, raw_scores, n_bins=10)

    # Plot calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration plot')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability in each bin')
    plt.legend()
    plt.title('Calibration Curve')
    plt.savefig('data/ROC_Curve_test_cali.png')
    plt.close()
    #save image

    # Apply calibration
    calibrated = CalibratedClassifierCV(estimator=best_model, method='sigmoid', cv='prefit')
    calibrated.fit(X_train, bool_train)  # You might need to split your training data for this step or use a different dataset
    calibrated_probs = calibrated.predict_proba(X_test)[:, 1]

    # Recalculate ROC AUC on calibrated probabilities
    fpr, tpr, thresholds = roc_curve(bool_test, calibrated_probs)
    roc_auc = auc(fpr, tpr)
    print('New AUC:', roc_auc)

    # Check if the model now outputs more realistic probabilities
    print("Sample probabilities:", calibrated_probs[:10])
    
    print("accuracy cali: ", accuracy_score(bool_test, calibrated.predict(X_test)))
    print("accuracy init :" , accuracy_score(bool_test,best_model.predict(X_test)))



if __name__ == "__main__":
    run_svm_analysis()





