#!/usr/bin/env python
# coding: utf-8

# # Handwritten Digit Recognition

# The objective of this project is to develop a machine learning-based multiclass classification system to recognize handwritten digits from 0 to 9 using numerical features extracted from pen movements.
# 
# The system uses the Pen-Based Recognition of Handwritten Digits dataset from the UCI Machine Learning Repository.
# 
# **The classification problem is implemented using five machine learning algorithms:**
# 
# - **Logistic Regression**
# - **Decision Tree Classifier**
# - **K-Nearest Neighbors (KNN)**
# - **Naive Bayes Classifier – Gaussian or Multinomial**
# - **Support Vector Machine (SVM)**

# #### Data Loading

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

warnings.filterwarnings("ignore")


# In[16]:


#load the dataset
train_df = pd.read_csv(
    "pendigits.tra",
    header=None
)

test_df = pd.read_csv(
    "pendigits.tes",
    header=None
)

print("Training shape:", train_df.shape)
print("Testing shape:", test_df.shape)


# In[17]:


# Define feature and target column names
feature_names = [f"feature_{i}" for i in range(1, 17)]
column_names = feature_names + ["digit"]

# Assign column names
train_df.columns = column_names
test_df.columns = column_names

# Separate input features and target variable
X_train = train_df[feature_names]
y_train = train_df["digit"]

X_test = test_df[feature_names]
y_test = test_df["digit"]

# Display dataset information
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Number of input features:", len(feature_names))
print("Training target classes:", sorted(train_df["digit"].unique()))
print("Testing target classes:", sorted(test_df["digit"].unique()))
print("\nMissing values in training data:", train_df.isnull().sum().sum())
print("Missing values in testing data:", test_df.isnull().sum().sum())


# ### Examine dataset size, missing values and features type

# In[18]:


# Examine dataset size, missing values and feature types

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

print("\nTraining data types:")
print(train_df.dtypes)

print("\nMissing values in training data:",
      train_df.isnull().sum().sum())

print("Missing values in testing data:",
      test_df.isnull().sum().sum())

print("\nNumber of input features:", X_train.shape[1])
print("Number of target classes:", y_train.nunique())


# ### 1. Logistic Regression

# In[19]:


lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=3000))
])
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)

print("Logistic Regression trained successfully!")


# In[20]:


print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("AUC Score:", roc_auc_score(
    y_test, 
    y_prob_lr,
    multi_class="ovr", 
    average="weighted"
))
print("Precision:", precision_score(
    y_test,
    y_pred_lr,
    average="weighted",
    zero_division=0    
))
print("Recall:", recall_score(
    y_test,
    y_pred_lr,
    average="weighted",
    zero_division=0
))
print("F1 Score:", f1_score(
    y_test, 
    y_pred_lr, 
    average="weighted",
    zero_division=0
))
print("MCC Score:", matthews_corrcoef(y_test,
                                      y_pred_lr))


# ### 2. Decision Tree Classifier

# In[21]:


dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)

print("Decision Tree Classifier")
print("Accuracy:", accuracy_score(y_test,
                                 y_pred_dt))
print("AUC Score:", roc_auc_score(
    y_test,
    y_prob_dt,
    multi_class="ovr",
    average="weighted"
))
print("Precision:", precision_score(
    y_test, 
    y_pred_dt,
    average="weighted",
    zero_division=0    
))
print("Recall:", recall_score(
    y_test,
    y_pred_dt,
    average="weighted",
    zero_division=0
))
print("F1 Score:", f1_score(
    y_test,
    y_pred_dt,
    average="weighted",
    zero_division=0
))
print("MCC Score:", matthews_corrcoef(
    y_test,
    y_pred_dt))


# ### 3. K-Nearest Neighbor Classifier

# In[22]:


knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)
y_prob_knn = knn_model.predict_proba(X_test)

print("K-Nearest Neighbor Classifier")
print("Accuracy:", accuracy_score(y_test,
                                 y_pred_knn))
print("AUC Score:", roc_auc_score(
    y_test,
    y_prob_knn,
    multi_class="ovr",
    average="weighted"
))
print("Precision:", precision_score(
    y_test,
    y_pred_knn,
    average="weighted",
    zero_division=0
))
print("Recall:", recall_score(
    y_test,
    y_pred_knn, 
    average="weighted",
    zero_division=0
))
print("F1 Score:", f1_score(
    y_test,
    y_pred_knn,
    average="weighted",
    zero_division=0
))
print("MCC Score:", matthews_corrcoef(y_test,
                                     y_pred_knn))


# ### 4. Naive Bayes Classifier - Gaussian

# In[23]:


nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)
y_prob_nb = nb_model.predict_proba(X_test)

print("Gaussian Naive Bayes Classifier")
print("Accuracy:", accuracy_score(y_test,
                                 y_pred_nb))
print("AUC Score:", roc_auc_score(
    y_test,
    y_prob_nb,
    multi_class="ovr",
    average="weighted"
))
print("Precision:", precision_score(
    y_test,
    y_pred_nb,
    average="weighted",
    zero_division=0
))
print("Recall:", recall_score(
    y_test,
    y_pred_nb, 
    average="weighted",
    zero_division=0
))
print("F1 Score:", f1_score(
    y_test,
    y_pred_nb,
    average="weighted",
    zero_division=0
))
print("MCC Score:", matthews_corrcoef(y_test,
                                     y_pred_nb))


# ### 5. Support Vector Machine

# In[24]:


svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ))
])

svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)

print("Support Vector Machine")
print("Accuracy:", accuracy_score(y_test,
                                 y_pred_svm))
print("AUC Score:", roc_auc_score(
    y_test,
    y_prob_svm,
    multi_class="ovr",
    average="weighted"
))
print("Precision:", precision_score(
    y_test, 
    y_pred_svm,
    average="weighted",
    zero_division=0
))
print("Recall:", recall_score(
    y_test,
    y_pred_svm,
    average="weighted",
    zero_division=0
))
print("F1 Score", f1_score(
    y_test, 
    y_pred_svm,
    average="weighted",
    zero_division=0
))
print("MCC Score", matthews_corrcoef(y_test,
                                    y_pred_svm))


# ### Comparison Table with the Evaluation Metrics 

# In[25]:


# Attribute names (column headings)
columns = [
    "",
    "ML Model Name",
    "Accuracy",
    "AUC Score",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC Score"
]

# Values (calculated results for each model)
data = [
    [
        1,
        "Logistic Regression",
        accuracy_score(y_test, y_pred_lr),
        roc_auc_score(y_test, y_prob_lr, multi_class="ovr",
                      average="weighted", labels=lr_model.classes_),
        precision_score(y_test, y_pred_lr, average="weighted", zero_division=0),
        recall_score(y_test, y_pred_lr, average="weighted", zero_division=0),
        f1_score(y_test, y_pred_lr, average="weighted", zero_division=0),
        matthews_corrcoef(y_test, y_pred_lr)
    ],
    [
        2,
        "Decision Tree",
        accuracy_score(y_test, y_pred_dt),
        roc_auc_score(y_test, y_prob_dt, multi_class="ovr",
                      average="weighted", labels=dt_model.classes_),
        precision_score(y_test, y_pred_dt, average="weighted", zero_division=0),
        recall_score(y_test, y_pred_dt, average="weighted", zero_division=0),
        f1_score(y_test, y_pred_dt, average="weighted", zero_division=0),
        matthews_corrcoef(y_test, y_pred_dt)
    ],
    [
        3,
        "K-Nearest Neighbors",
        accuracy_score(y_test, y_pred_knn),
        roc_auc_score(y_test, y_prob_knn, multi_class="ovr",
                      average="weighted", labels=knn_model.classes_),
        precision_score(y_test, y_pred_knn, average="weighted", zero_division=0),
        recall_score(y_test, y_pred_knn, average="weighted", zero_division=0),
        f1_score(y_test, y_pred_knn, average="weighted", zero_division=0),
        matthews_corrcoef(y_test, y_pred_knn)
    ],
    [
        4,
        "Gaussian Naive Bayes",
        accuracy_score(y_test, y_pred_nb),
        roc_auc_score(y_test, y_prob_nb, multi_class="ovr",
                      average="weighted", labels=nb_model.classes_),
        precision_score(y_test, y_pred_nb, average="weighted", zero_division=0),
        recall_score(y_test, y_pred_nb, average="weighted", zero_division=0),
        f1_score(y_test, y_pred_nb, average="weighted", zero_division=0),
        matthews_corrcoef(y_test, y_pred_nb)
    ],
    [
        5,
        "Support Vector Machine",
        accuracy_score(y_test, y_pred_svm),
        roc_auc_score(y_test, y_prob_svm, multi_class="ovr",
                      average="weighted", labels=svm_model.classes_),
        precision_score(y_test, y_pred_svm, average="weighted", zero_division=0),
        recall_score(y_test, y_pred_svm, average="weighted", zero_division=0),
        f1_score(y_test, y_pred_svm, average="weighted", zero_division=0),
        matthews_corrcoef(y_test, y_pred_svm)
    ]
]

# Create table using separate columns and data
results_df = pd.DataFrame(data, columns=columns)

# Display calculated values
display(results_df.round(4)
       .style
    .hide(axis="index")
   .set_table_styles([
        {
            "selector": "th",
            "props": [("font-weight", "bold")]
        },
        {
            "selector": "td.col0",
            "props": [("font-weight", "bold")]

        }
       ,
        {
            "selector": "td.col1",
            "props": [("font-weight", "bold"), ("text-align", "left")]
        }
    ])
       )


# In[26]:


import os
import joblib

# Create model directory
os.makedirs("model", exist_ok=True)

# Save all five trained models
joblib.dump(lr_model, "model/logistic_regression.pkl")
joblib.dump(dt_model, "model/decision_tree.pkl")
joblib.dump(knn_model, "model/knn.pkl")
joblib.dump(nb_model, "model/naive_bayes.pkl")
joblib.dump(svm_model, "model/svm.pkl")

print("All five models saved successfully!")


# In[27]:


model_directory = "model"

print("Saved Model Files:")

for file in os.listdir(model_directory):
    print(file)


# In[28]:


logistic_model = joblib.load("model/logistic_regression.pkl")
decision_tree_model = joblib.load("model/decision_tree.pkl")
knn_model = joblib.load("model/knn.pkl")
naive_bayes_model = joblib.load("model/naive_bayes.pkl")
svm_model = joblib.load("model/svm.pkl")

print("All five models loaded successfully!")


# In[ ]:





# In[ ]:





# In[ ]:




