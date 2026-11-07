# Machine Learning Assignment 2 — Classification Model Comparison

**Programme:** BITS Pilani, M.Tech (Artificial Intelligence and Machine Learning)  
**Course:** Machine Learning — Assignment 2

## a. Problem Statement

The objective is to implement and compare five supervised machine learning classification algorithms using the **same publicly available dataset**. Each model is trained on the training subset and evaluated on an unseen test subset using six metrics: Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The project also includes an interactive Streamlit application that accepts a labeled test CSV, allows users to select a trained model, and displays metrics, a confusion matrix, and a classification report **without retraining on test data**.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic); loaded through `sklearn.datasets.load_breast_cancer`
- **Instances:** 569
- **Input features:** 30 numerical measurements
- **Target:** Binary classification — benign or malignant
- **Class coding in this project:** malignant = 1 (positive class), benign = 0
- **Training/testing:** 80% / 20%, stratified split, `random_state=42` (455 training and 114 testing records)
- **Preprocessing:** StandardScaler is fitted only on the training data within pipelines for Logistic Regression, KNN, and SVM.

The dataset meets the assignment's minimum requirements of 500 instances and 12 features. This project is educational and is **not a medical diagnostic system**.

## c. GitHub Repository Link

**GitHub:** https://github.com/2026ak05296-cloud/BITS-ML-Assignment-2

The repository contains source code, the notebook, `requirements.txt`, training/test CSV files, configuration, and the comparison results. **Before submitting, ensure the `model/` folder contains all five saved trained classifiers.**

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

### Interpretation and Limitations

The malignant class is the positive class, so recall measures how many actual malignant samples were detected. In an educational medical-themed dataset, false negatives deserve special attention. These observations refer only to this held-out test dataset; cross-validation and additional datasets would be needed to examine generalization.

## Streamlit Application Link

**New Streamlit app link:** ADD YOUR NEW STREAMLIT DEPLOYMENT URL HERE (the previous link belongs to the earlier deployment).

**Before submission:** Deploy this repository on Streamlit Community Cloud and update the URL above.

### Streamlit Features

1. Upload `test_data.csv` with the 30 feature columns and `target` column.
2. Choose one of the five pretrained models from the dropdown.
3. View Accuracy, AUC, Precision, Recall, F1, and MCC.
4. View the confusion matrix and classification report.
5. Evaluate using saved models without fitting on the uploaded test data.

## Reproduce the Experiment

1. Install dependencies: `pip install -r requirements.txt`
2. Run `python train_models.py` in the **BITS Virtual Lab**, or execute the notebook `ML_Assignment_2.ipynb` with the training script available in the working directory.
3. Confirm that `model_comparison.csv`, `test_data.csv`, `model_config.json`, and all five files in `model/` were produced.
4. Start the application using `streamlit run app.py` and upload the generated `test_data.csv`.
5. Check that the metrics match the recorded results, accounting for any library-version differences.

**Mandatory evidence:** Capture one authentic screenshot showing successful execution in **BITS Virtual Lab** and include it in the final submission PDF alongside the GitHub link, Streamlit URL, and this README.

## Academic Integrity

This README is a structured draft based on the repository's recorded outputs. Re-run the experiment, review the analysis, customize the presentation, and submit only results and evidence that you have personally verified.
