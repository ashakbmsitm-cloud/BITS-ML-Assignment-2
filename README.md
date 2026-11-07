## a. Problem Statement

The objective is to implement and compare five supervised machine learning classification algorithms using the **same publicly available dataset**. Each model is trained on the training subset and evaluated on an unseen test subset using six metrics: Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic); loaded through `sklearn.datasets.load_breast_cancer`
- **Instances:** 569
- **Input features:** 30 numerical measurements
- **Target:** Binary classification — benign or malignant
- **Class coding in this project:** malignant = 1 (positive class), benign = 0
- **Training/testing:** 80% / 20%, stratified split, `random_state=42` (455 training and 114 testing records)
- **Preprocessing:** StandardScaler is fitted only on the training data within pipelines for Logistic Regression, KNN, and SVM.

## c. GitHub Repository Link

**GitHub:** https://github.com/ashakbmsitm-cloud/BITS-ML-Assignment-2

## d. Models Used and Evaluation Metrics

The following values are from `model_comparison.csv` in this repository, rounded to four decimal places. They should be checked against your own execution in BITS Virtual Lab.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

## Observations on Model Performance

The observations below are **draft interpretations** of the recorded scores. Review and express them in your own words after reproducing the results.

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Achieved high accuracy (0.9649) and the **highest AUC (0.9960)**, suggesting strong ranking of the two classes. Its malignant recall (0.9286) was equal to SVM, but its F1 score was slightly lower. |
| **Decision Tree** | Had the lowest accuracy (0.9211), F1 (0.8861), and MCC (0.8299) among these models. Recall of 0.8333 indicates that it missed more malignant cases than Logistic Regression, KNN, or SVM on this test split. |
| **KNN** | Performed well (accuracy 0.9561, F1 0.9383). Feature scaling is important because KNN relies on distances. Its malignant recall (0.9048) was below Logistic Regression and SVM. |
| **Naive Bayes** | Achieved precision of 1.0000 for malignant predictions, but recall was only 0.8333, meaning some malignant examples were missed. The Gaussian feature-independence assumption may limit this model. |
| **Support Vector Machine** | Delivered the **highest accuracy (0.9737), F1 (0.9630), and MCC (0.9442)**, along with precision of 1.0000 and recall of 0.9286. Its AUC (0.9947) was slightly below Logistic Regression. |
| **Overall Winner for This Dataset** | **Support Vector Machine (SVM)** is preferred on this held-out test split because it leads in accuracy, F1, and MCC while maintaining high malignant recall. However, Logistic Regression leads in AUC; results from a single split do not establish universal superiority. |

## Streamlit Application Link

https://bits-ml-assignment-2-jgnltcuzawmgj9rlutrxbt.streamlit.app/

