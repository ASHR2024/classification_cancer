AIAP Batch 16 Technical Assessment\
Name: Regina Ang Swee Hoon\
email: angshregina@gmail.com


1. Executive Summary
    
    The objective of studying this dataset is to predict the occurence of lung cancer in patients based on several physical and lifestyle attributes.

    Data exploratory is first performed to understand the data and determine useful features to conduct a classification model.

    We build an end-to-end Machine Learning Pipeline to guide the preprocessing stage followed by understanding how the classification models are chosen and evaluated. Several base models are run before selecting the best model to perform hyperparameter tuning.

2. Folder Structure
  
    \- .github/workflows\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;github-actions.yml\
  &nbsp;- src\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; _pycache\
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; func.cpython-310.pyc\
&nbsp;&nbsp;&nbsp;&nbsp;app_cancer.py\
  &nbsp;&nbsp;&nbsp;&nbsp;func.py\
  &nbsp;.DS_Store\
    &nbsp;&nbsp;EDA.ipynb\
  &nbsp;&nbsp;README.md\
  &nbsp;&nbsp;requirements.txt\
  &nbsp;&nbsp;run.sh

3. Instruction

    For Linux\
    With Python activated in your environment:

    Clone this project\
    git clone https://github.com/ASHR2024/aiap16-ang-swee-hoon-regina-816Z.git

    Run the shell bash script:\
    bash run.sh

    For Windows\
    Open a python shell in your terminal:

    Clone this project\
    git clone https://github.com/ASHR2024/aiap16-ang-swee-hoon-regina-816Z.git

    Run the shell bash script:\
    bash run.sh


4. Project Pipeline Flow

    &nbsp;&nbsp;&nbsp;&nbsp;4.1 Data Retrieval\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Data source is imported using SQLite using the relative path 'data/lung_cancer.db'.\
    &nbsp;&nbsp;&nbsp;&nbsp;4.2 Data Preprocessing\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Handle missing values, inconsistent labels and distribution of the attributes.   
    &nbsp;&nbsp;&nbsp;&nbsp;4.3 Feature Engineering\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Perform one-hot encoding on categorical values, create new features from existing ones to provide additional predictive power.\
    &nbsp;&nbsp;&nbsp;&nbsp;4.4 Data Splitting\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Split the dataset into train and test set. The train set is used to train the models and the test set is used to assess the models' performance on unseen data. \
    &nbsp;&nbsp;&nbsp;&nbsp;4.5 Model Selection\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Compare the performance of all the models and use evaluation metrics (accuracy, precision, recall, F1-score) to select the best model.\
    &nbsp;&nbsp;&nbsp;&nbsp;4.6 Hyperparameter Tuning\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- RandomizedSearchCV is used to search for the best hyperparameters for the chosen model (best base model).\
    &nbsp;&nbsp;&nbsp;&nbsp;4.7 Model Selection and Final Evaluation\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Assess the final selected model using the best hyperparameters obtained from the previous step to first train the model followed by testing it with test set.\
    &nbsp;&nbsp;&nbsp;&nbsp;4.8 Documentation and Reporting\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Explanation and justification provided on each step of the process described above.


5. Overview of key findings

    &nbsp;&nbsp;&nbsp;&nbsp;5.1 EDA and key findings\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Missing values found in "COPD History" and "Taken Bronchodilators"\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Negative values in Age\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Gender values are not consistent with one observation having "NAN" as value\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Not all independent variables are equally distributed although the dependent variable (Lung Cancer Occurence) has an acceptable distribution\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-Irrelevant features to be excluded: ID, Last Weight, Current Weight, Start Smoking, Stop Smoking, Dominant Hand

    &nbsp;&nbsp;&nbsp;&nbsp;5.2 Feature Engineering\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Create a new feature "Avg Weight" to replace "Last Weight" and "Current Weight" as the variance in the weight difference is rather constant.\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Two new features are created to replace "Start Smoking" year and "Stop Smoking" year as count of years with respect to current year (2024).\
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Perfom one-hot encoding to the categorical features: 'Gender', 'COPD History', 'Genetic Markers', 'Air Pollution Exposure', 'Taken Bronchodilators', 'Frequency of Tiredness' and then reduce each feature by one column

6. Features that are processed
    Feature | Description | Processed
    ---|---|---
    Age | Age of the patient | Remove observations containing negative value
    Gender | Gender of the patient | Standardized MALE to Male; FEMALE to Female
    Last Weight | Last Officially recorded weight of patient | used to calculate the average weight
    Current Weight | LCurrent Officially recorded weight of patient | used to calculate the average weight
    Start Smoking | Year that the patient starts smoking | Created new feature called "Smoke_Years" where the years are converted to number of years where "Not Applicable" is converted to 0; Year 2024 is set to 1; preceding years will add 1 accordingly
    Stop Smoking | Year that the patient stops smoking | Created new feature called "Stop_Years" where the years are converted to number of years where "Not Applicable" is converted to 0; "Still Smoking" is first set to "2024" before setting it to 1; preceding years will add 1 accordingly
  
7. Choice of Models

    Logistic Regression (LR) - assumes a linear relationship between the features and log-odds of the target variable. Well suited for binary classification problems. It also assumes observations are independent of each other.

    Gaussian Naive Bayes (GNB) -  a probabilisitc classification algorithm that assumes that features are continuous and independent of each other, which may not be suitable for this dataset but it works well with high-dimensional data so it is worth a try.

    Support Vector Classifier (SVC) - can handle both linear and non-linear decision boundaries using different kernel functions (e.g., linear, polynomial, radial basis function). Does not make strong assumptions about the underlying data distribution and it works well with small to medium-sized datasets.

    Decision Tree Classifier (DT) - is a non-parametric algorithm (does not require that data follow a normal distribution). Can handle both numerical and categorical data.

8. Evaluation of Models

    Logistic Regression \
Precision: Precision for class 0 is 0.58, meaning that among all instances predicted as class 0, 58% are actually class 0.
Precision for class 1 is 0.62, meaning that among all instances predicted as class 1, 62% are actually class 1.
Recall: Recall for class 0 is 0.48, meaning that among all actual instances of class 0, 48% are correctly predicted as class 0.
Recall for class 1 is 0.70, meaning that among all actual instances of class 1, 70% are correctly predicted as class 1.
F1-score: F1-score for class 0 is 0.53, indicating a moderate balance between precision and recall for class 0.
F1-score for class 1 is 0.66, indicating a balance between precision and recall for class 1.
Accuracy: Overall accuracy of 60 means the model correctly classifies 60% of the instances in the dataset.

    Gaussian Naive Bayes\
Precision: Precision for class 0 is 0.57, meaning that among all instances predicted as class 0, 57% are actually class 0.
Precision for class 1 is 0.62, meaning that among all instances predicted as class 1, 62% are actually class 1.
Recall: Recall for class 0 is 0.51, meaning that among all actual instances of class 0, 51% are correctly predicted as class 0.
Recall for class 1 is 0.67, meaning that among all actual instances of class 1, 67% are correctly predicted as class 1.
F1-score: F1-score for class 0 is 0.54, indicating a moderate balance between precision and recall for class 0.
F1-score for class 1 is 0.64, indicating a balance between precision and recall for class 1.
Accuracy: Overall accuracy of 60 means the model correctly classifies 60% of the instances in the dataset.

    Support Vector Classifier\
Precision: Precision for class 0 is 0.67, meaning that among all instances predicted as class 0, 67% are actually class 0.
Precision for class 1 is 0.72, meaning that among all instances predicted as class 1, 72% are actually class 1.
Recall: Recall for class 0 is 0.67, meaning that among all actual instances of class 0, 67% are correctly predicted as class 0.
Recall for class 1 is 0.73, meaning that among all actual instances of class 1, 73% are correctly predicted as class 1.
F1-score: F1-score for class 0 is 0.67, indicating a balance between precision and recall for class 0.
F1-score for class 1 is 0.72, indicating a good balance between precision and recall for class 1.
Accuracy: Overall accuracy of 70 means the model correctly classifies 70% of the instances in the dataset.

    Decision Tree Classifier\
Precision: Precision for class 0 is 0.61, meaning that among all instances predicted as class 0, 61% are actually class 0.
Precision for class 1 is 0.68, meaning that among all instances predicted as class 1, 68% are actually class 1.
Recall: Recall for class 0 is 0.64, meaning that among all actual instances of class 0, 64% are correctly predicted as class 0.
Recall for class 1 is 0.65, meaning that among all actual instances of class 1, 65% are correctly predicted as class 1.
F1-score: F1-score for class 0 is 0.62, indicating a balance between precision and recall for class 0.
F1-score for class 1 is 0.67, indicating a balance between precision and recall for class 1.
Accuracy: Overall accuracy of 65 means the model correctly classifies 65% of the instances in the dataset.

    Hyperparameter tuning on SVC\
The accuracy scores of the train and test set are 0.699 and 0.697 respectively, close to the accuracy score of SVC base model with 0.70.

9. Conclusion

SVC is the best choice as it generates the best accuracy without the need for hyperparameter tuning. Furthermore, it has the highest precision and recall for class 1 meaning patients with cancer are correctly identified or classified.

    DT can be the next preferred model as the accuracy is slightly lower than SVC. The lower precision for class 1 may also signify a higher false positive (FP) which in the sense of cancer detection, is a safe risk.

    Due to time constraint, the parameters for DT and hyperparameter tuning on SVC are not further explored, which could potentially lead to better model performance.