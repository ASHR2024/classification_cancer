import pandas as pd
import numpy as np
from func import tr_tst_spl, regress_models

# Databases
import sqlite3

#-------------------connect to data----------------------#
try:
    con = sqlite3.connect('../data/lung_cancer.db')
except Exception as err:
    print(f"Connection error:\n{err}")

# Extract data as pd.DataFrame
cur = con.cursor()
raw_df = pd.read_sql_query('SELECT * FROM lung_cancer', con)

# Close db connection
con.close()

print(raw_df)

#--------------------------Preprocessing-----------------------#
df = raw_df.copy()

#remove null values
df = df.dropna()

# New df without the negative Age values
df = df[df['Age'] > 0]

# Standardise Gender category as "Male" and "Female"
df['Gender'].replace('MALE', 'Male', inplace=True)
df['Gender'].replace('FEMALE', 'Female', inplace=True)

# Check if any row contains the "NAN" and remove them if present
if (df['Gender'].str.contains("NAN")).any():
    # Update the DataFrame by filtering out rows containing the search string
    df = df[~df['Gender'].str.contains("NAN")]

# Create a new feature "Avg Weight" in lieu of "Last Weight" and "Current Weight"
df['Avg Weight']= (df['Last Weight'] + df['Current Weight'])/2

# -----------------Preprocess "Start Smoking" and "Stop Smoking" ----------------#
import datetime

# convert "Not Applicable" to 0
# convert "Start Smoking" years to int to calculate no. of years
df['Start Smoking'] = df['Start Smoking'].replace("Not Applicable", 0)
df['Start Smoking'] = df['Start Smoking'].astype(int)

# replace "Not Applicable" to 0; "Still smoking" to current year
# convert "Stop Smoking" years to int to calculate no. of years

df['Stop Smoking'] = df['Stop Smoking'].replace("Not Applicable", 0)
df['Stop Smoking'] = df['Stop Smoking'].replace("Still Smoking", datetime.datetime.now().year)
df['Stop Smoking'] = df['Stop Smoking'].astype(int)

# Year 2024 set as 1 and step 1 with each preceding year
# "Not Applicable" will be set as 0

# Get the current year
current_year = datetime.datetime.now().year

# Create new features to replace "Start Smoking" and "Stop Smoking"
df['Smoke_Years'] = df['Start Smoking'].apply(lambda x: (current_year - x + 1) if x!=0 else 0)
df['Stop_Years'] = df['Stop Smoking'].apply(lambda x: (current_year - x + 1) if x!=0 else 0)


# ------------ One-hot encoding to the categorical attributes ----------------------#

X_enc = pd.get_dummies(df, columns = ['Gender', 'COPD History','Genetic Markers','Air Pollution Exposure',
                                      'Taken Bronchodilators','Frequency of Tiredness'])

#Remove all irrelevant columns
X_enc = X_enc.drop(["ID", "Last Weight","Current Weight", "Start Smoking", "Stop Smoking", "Dominant Hand", "Gender_Female", 
            "COPD History_No", "Genetic Markers_Present", "Air Pollution Exposure_Medium","Taken Bronchodilators_No",
           "Frequency of Tiredness_High" ], axis=1)

print(X_enc)

#--------------------Perform scaling-----------------------------#
from sklearn.preprocessing import StandardScaler
y = X_enc['Lung Cancer Occurrence']
X_enc.drop(['Lung Cancer Occurrence'], axis=1, inplace=True)
sc = StandardScaler()
X_sc = sc.fit_transform(X_enc)

# -----------Split train and test set for other base models-----------
rand_seed = 42
ts = 0.20
X_train, X_test, y_train, y_test = tr_tst_spl(X_sc, y, ts, rand_seed)

#---------run base models-----------------#
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

regressors = [LogisticRegression(), GaussianNB(), SVC(),
              DecisionTreeClassifier(min_samples_split = 10, min_samples_leaf = 10)]

regress_models (regressors,X_train, X_test, y_train, y_test)

#--------------Hyperparameter Tuning using GridSearchCV On SVC (Best Model)------#
#Hypertuning on the selected model SVC

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

parameters = {
    'C': uniform(loc=0, scale=10),  # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
    'gamma': ['scale', 'auto'],  # Kernel coefficient
}

# Perform RandomizedSearchCV
from time import time

model = SVC()
rand_search = RandomizedSearchCV(estimator=model, param_distributions=parameters, n_iter=100, cv=5, random_state=42)
start = time()
rand_search.fit(X_train, y_train)
train_time = time() - start
print("Training time: %0.3fs" % train_time)
print("train accuracy",rand_search.score(X_train, y_train))

# Get the best hyperparameters
print("Best params",rand_search.best_params_)

# Evaluate the model with the best hyperparameters
start = time()
best_model = rand_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
predict_time = time()-start
print("Prediction time: %0.3fs" % predict_time)
print("test accuracy:", accuracy)