# 📌 Employee Absenteeism Prediction using Logistic Regression

This project predicts employee absenteeism using logistic regression. The solution is built with a clear focus on **model interpretability**, **business relevance**, and **threshold optimization** to maximize actionable insights for HR and management.

---

## 🧱 Project Structure

This repository includes four Jupyter Notebooks that walk through the entire model-building lifecycle:

| Notebook                     | Purpose                                      |
|------------------------------|----------------------------------------------|
| `1_data_preprocessing.ipynb` | Data cleaning and feature engineering        |
| `2_baseline_model.ipynb`     | Baseline logistic regression model           |
| `3_model_improvements.ipynb` | Regularization and threshold tuning          |
| `4_final_model.ipynb`        | Final model with optimal threshold + export  |

---

## 📊 Dataset

- The dataset (`company_employee_data.csv`) contains employee records including absence reasons, demographics, and work-related metrics.
- The goal is to predict whether an employee will be **excessively absent** (i.e., above the dataset’s median absence hours).

---

## 🧹 Data Preprocessing Summary

In this section, we preprocessed the data by performing the following steps:

- Dropped irrelevant columns (e.g., `ID`) to eliminate noise from the data.
- One-hot encoded the `Reason for Absence` column using `pd.get_dummies()` to convert categorical data into numerical format.
- Grouped reasons into 4 types (`Reason_1` to `Reason_4`) based on column index ranges and reasons for absence image, grouping similar reasons into one class.
- Simplified and extracted `Month` and `Day of the Week` from the date column.
- Mapped `Education` levels to binary (0 = basic, 1 = higher education).
- Reordered and renamed columns for better readability and model compatibility.
- Created a final preprocessed DataFrame ready for analysis or machine learning modeling.
- Saved as `employee_preprocessed_data.csv`.

---

## 📈 Baseline Logistic Regression Summary

### Model Development:

- Created a binary target column `Excessive Absenteeism` using median of hours as cutoff.
- Applied `CustomScaler` to selectively scale numerical columns.
- Performed train-test split (80/20) using `train_test_split`.
- Trained a logistic regression model using `sklearn`.

### Evaluation Metrics:

- Used classification report (Precision, Recall, F1-score).
- Calculated ROC AUC score.
- Extracted model coefficients and odds ratios for interpretation.

**Baseline Metrics:**

- ROC AUC: `0.792`
- Recall (Class 1): `0.70`

---

## 🔍 Model Improvements

### Regularization:

- Applied **L1 (Lasso)** and **L2 (Ridge)** regularization.
- No significant improvements observed because:
  - Dataset is balanced.
  - No signs of overfitting.

### Threshold Tuning:

- Plotted **Precision vs Recall** for various thresholds.
- Identified **threshold = 0.42** offers:
  - Slightly better recall
  - Balanced precision
  - Improved business-aligned prediction

---

## ✅ Final Model Summary

- Used all features (did not drop low-impact columns).
- Applied optimal threshold = `0.42`.
- Trained using logistic regression and evaluated on test data.

### Final Metrics:

| Metric             | Baseline | Final Model |
|--------------------|----------|-------------|
| ROC AUC            | 0.792    | 0.800       |
| Recall (Class 1)   | 0.70     | 0.77        |
| F1-Score           | 0.72     | 0.73        |

---

## 🎯 Business Relevance

- **Higher recall** improves ability to detect likely absentees (Class 1).
- **Better F1-score** and **ROC AUC** show stronger overall performance.
- **Threshold tuning** allows better trade-off between precision and recall.
- Ideal for use in HR systems or dashboards to flag high-risk employees.

---

## 💾 Model Deployment

We have saved both the trained model and the scaler:

- `final_logistic_model.pkl` – the trained logistic regression model.
- `final_scaler.pkl` – the fitted `StandardScaler` for data standardization.

---

## 📌 How to Use

1. Preprocess your new dataset to match the structure of the training data.
2. Load the model and scaler:

```python
import pickle
model = pickle.load(open('final_logistic_model.pkl', 'rb'))
scaler = pickle.load(open('final_scaler.pkl', 'rb'))
