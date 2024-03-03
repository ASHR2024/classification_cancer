import pandas as pd
import numpy as np

#-------------train test split----------------------------------#
from sklearn.model_selection import train_test_split

# Split into training and testing
def tr_tst_spl(x, y, ts, rand_seed):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = ts,
                                                    random_state = rand_seed,
                                                    shuffle = True,
                                                    stratify = y)
    print(f'\nTraining set size: {len(X_train)}\n'
          f'Testing set size: {len(X_test)}\n')
    
    return X_train, X_test, y_train, y_test

#---------running models-----------#
from time import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

import matplotlib.pyplot as plt

#print confusion matrix guide
def cm_guide():
    print("[TN  FP]")
    print("[FN  TP]")


def regress_models (regressors,X_train, X_test, y_train, y_test):
    for model in regressors:
        start = time()
        model.fit(X_train, y_train)
        train_time = time() - start
        start = time()
        y_pred = model.predict(X_test)
        predict_time = time()-start  

        print(model)
        print("Training time: %0.3fs" % train_time)
        print("Prediction time: %0.3fs" % predict_time)

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Calculate classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Print confusion matrix and classification metrics
        print("\nConfusion Matrix:")
        cm_guide()
        print(cm)
        print("\nAccuracy: %0.2f" % accuracy)
        print("Precision: %0.2f" % precision)
        print("Recall: %0.2f" % recall)
        print("F1 Score: %0.2f" % f1)

        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
      