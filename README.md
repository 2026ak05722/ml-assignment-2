# Handwritten Digit Recognition Using Machine Learning

## a. Problem Statement

The objective of this project is to develop a machine learning-based multiclass classification system to recognize handwritten digits from 0 to 9 using numerical features extracted from pen movements.

The system uses the **Pen-Based Recognition of Handwritten Digits** dataset from the UCI Machine Learning Repository.

Five machine learning classification algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The performance of each model is evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

## b. Dataset Description

**Dataset Name:** Pen-Based Recognition of Handwritten Digits

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits

### Dataset Characteristics

| Attribute | Description |
|---|---|
| Total instances | 10,992 |
| Training instances | 7,494 |
| Testing instances | 3,498 |
| Total input features | 16 |
| Feature type | Numerical |
| Target variable | digit |
| Number of classes | 10 |
| Target classes | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| Missing values | 0 |
| Classification type | Multiclass |

### Dataset Files

- `pendigits.tra` – Training dataset.
- `pendigits.tes` – Testing dataset.
- `pendigits.names` – Dataset description and feature information.

The dataset contains 16 numerical attributes representing pen-coordinate measurements. The target variable represents the handwritten digit.

The official predefined training and testing datasets are used without creating an additional random split.



## c. GitHub Repository Link

**GitHub Repository:** https://github.com/2026ak05722/ml-assignment-2

## d. Models Used and Evaluation Metrics

All five classification algorithms are trained using the same training dataset and evaluated using the same independent testing dataset.

### Machine Learning Algorithms

1. **Logistic Regression:** A linear classification algorithm used for multiclass classification.
2. **Decision Tree Classifier:** Uses decision rules to classify handwritten digit patterns.
3. **K-Nearest Neighbors (KNN):** Classifies digits based on similarity between feature measurements.
4. **Gaussian Naive Bayes:** Uses probability distributions and the feature independence assumption.
5. **Support Vector Machine (SVM):** Uses an RBF kernel to identify decision boundaries between digit classes.

### Evaluation Metrics

The following six evaluation metrics are calculated:

1. **Accuracy:** Measures the proportion of correctly classified predictions.
2. **AUC Score:** Measures the model's ability to distinguish between classes using One-vs-Rest multiclass ROC AUC.
3. **Precision:** Measures the correctness of positive class predictions.
4. **Recall:** Measures the ability to identify actual class instances.
5. **F1 Score:** Harmonic mean of Precision and Recall.
6. **MCC Score:** Measures the correlation between actual and predicted classifications.

Weighted averaging is used for Precision, Recall, F1 and AUC.

### Model Comparison Table

The following table presents the evaluation results obtained from the testing dataset.

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.916000 | 0.990400 | 0.920400 | 0.916000 | 0.916400 | 0.907000 |
| Decision Tree | 0.917700 | 0.954200 | 0.919100 | 0.917700 | 0.917400 | 0.908700 |
| K-Nearest Neighbors | 0.976000 | 0.994000 | 0.976200 | 0.976000 | 0.976000 | 0.973300 |
| Gaussian Naive Bayes | 0.822500 | 0.964600 | 0.827900 | 0.822500 | 0.818000 | 0.804500 |
| **Support Vector Machine** | **0.979100** | **0.999600** | **0.979500** | **0.979100** | **0.979100** | **0.976900** |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved 91.60% accuracy. It provides good classification performance but performs below KNN and SVM. |
| Decision Tree | Achieved 91.77% accuracy. It performs slightly better than Logistic Regression but has a comparatively lower AUC score of 0.9542. |
| K-Nearest Neighbors | Achieved 97.60% accuracy. It provides strong classification performance and is the second-best model across the evaluation metrics. |
| Gaussian Naive Bayes | Achieved 82.25% accuracy. It has the lowest accuracy among the five models, potentially due to its feature independence assumption. |
| Support Vector Machine | Achieved 97.91% accuracy and 0.9996 AUC. It provides the best overall classification performance among the five evaluated models. |
| **Overall Winner** | **Support Vector Machine (SVM)** achieved the highest Accuracy, AUC, Precision, Recall, F1 Score and MCC Score. |

