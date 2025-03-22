# Employee_absentism_prediction_using_logistic_regression

This project predicts employee absenteeism using logistic regression. The solution is built with a clear focus on model interpretability, business relevance, and threshold optimization to maximize actionable insights for HR and management.

🧱 Project Structure

This repository includes four Jupyter Notebooks that walk through the entire model-building lifecycle:

Notebook Purpose
1_data_preprocessing.ipynb	Data cleaning and feature engineering
2_baseline_model.ipynb	Baseline logistic regression model
3_model_improvements.ipynb	Regularization and threshold tuning
4_final_model.ipynb	Final model with optimal threshold + export

📊 Dataset

The dataset (company_employee_data.csv) contains employee records including absence reasons, demographics, and work-related metrics.
The goal is to predict whether an employee will be excessively absent (above the dataset’s threshold value).

🧹 1. Data Preprocessing

Key steps:
	•	Dropped irrelevant columns (e.g., ID, nominal columns)
	•	One-hot encoded Reason for Absence, grouped into 4 reason types (Reason_1 to Reason_4)
        • Simplified and extracted month and weekday values from time column 
	•	Simplified Education into binary (basic vs higher)
	•	Reordered and renamed columns for clarity
	•	Created final cleaned dataframe for modeling (saved as **employee_preprocessed_data.csv**)

📈 2. Baseline Logistic Regression

Model development included:
	•	Created binary target (Excessive Absenteeism) using median split
	•	Applied custom feature scaling using CustomScaler
	•	Split data into train/test (80/20)
	•	Trained logistic regression using sklearn
	•	Evaluated using:
	•	Classification Report
	•	ROC AUC
	•	Feature Coefficients + Odds Ratios

Baseline ROC AUC: 0.792
Recall for absentees (Class 1): 0.70

🔍 3. Model Improvements

Regularization:
	•	Tried both L1 and L2 regularized logistic regression
	•	Minimal performance improvement due to:
	•	Balanced dataset
	•	No significant overfitting

Threshold Tuning:
	•	Plotted precision vs recall for different thresholds
	•	Identified that threshold of 0.42 offers:
	•	Better recall
	•	Balanced precision
	•	Business-aligned predictions

✅ 4. Final Model

Final model details:
	•	Trained on all features (no dropped columns)
	•	Applied custom threshold = 0.42
	•	Exported model and scaler for deployment

Final Model Metrics:

Metric	          Baseline	Final Model
ROC AUC           0.792   	0.800
Recall (Class 1)	0.70	    0.77
F1-Score	        0.72	    0.73

🎯 Business Relevance: Our use case focuses on identifying absentees.
The final model improves recall significantly without compromising precision.

💾 Model Deployment
	•	Exported model as final_logistic_model.pkl
	•	Exported custom scaler as final_scaler.pkl
	•	Easily reusable for batch prediction or integration into HR systems

📌 How to Use
	1.	Preprocess your new dataset (match feature structure)
	2.	Load model and scaler:

import pickle
model = pickle.load(open('final_logistic_model.pkl', 'rb'))
scaler = pickle.load(open('final_scaler.pkl', 'rb'))

  3.	Scale your inputs and predict:
  X_scaled = scaler.transform(new_data)
  predictions = model.predict_proba(X_scaled)[:, 1]


