# Heart-Disease-Prediction
Heart Disease Prediction
📌 Project Overview

Heart Disease Prediction is a machine learning project that predicts the likelihood of heart disease in a patient based on medical attributes. The goal is to build and compare multiple classification models and evaluate their performance using appropriate healthcare-focused metrics.

This project is intended as a decision-support system, not a replacement for medical professionals.

🧠 Problem Statement

Heart disease is one of the leading causes of death worldwide. Early prediction using clinical data can help in timely diagnosis and preventive care. This project applies machine learning techniques to analyze patient data and predict the presence of heart disease.

📊 Dataset

Dataset used: UCI Heart Disease Dataset

Each record represents a patient with medical attributes.

Features include:

age, sex

cp (chest pain type)

trestbps (resting blood pressure)

chol (cholesterol)

fbs (fasting blood sugar)

restecg

thalach (maximum heart rate achieved)

exang (exercise-induced angina)

oldpeak

slope, ca, thal

Target Variable:

0 → No heart disease

1 → Heart disease present

⚙️ Data Preprocessing

Handling missing values

Encoding categorical variables

Feature scaling (important for Logistic Regression)

Train-test split for model evaluation

Proper preprocessing ensures better model performance and fairness in comparison.

🤖 Machine Learning Models Used

Logistic Regression

Baseline model

Simple and interpretable

Decision Tree Classifier

Captures non-linear relationships

Easy to understand but prone to overfitting

Random Forest Classifier

Ensemble model

Reduces overfitting

Generally provides the best performance

Multiple models are used to compare accuracy, robustness, and generalization.

📈 Model Evaluation

The models are evaluated using the following metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

ROC–AUC Score

Why ROC–AUC?

In healthcare, false negatives are critical. ROC–AUC helps evaluate how well the model distinguishes between patients with and without heart disease.

📉 Visualizations

Confusion Matrix heatmaps

ROC curves for model comparison

Accuracy comparison between models

Feature importance (Random Forest)

These visualizations help interpret model performance clearly.

🏥 Healthcare Perspective

High recall is prioritized to minimize missed disease cases.

The model is designed as a support tool for clinicians, not a diagnostic authority.

Predictions should always be interpreted alongside medical expertise.

⚠️ Limitations

Dataset size is limited

Model is not clinically validated

Performance may vary with real-world hospital data

Requires expert validation before real deployment

🛠️ Tech Stack

Python

NumPy

Pandas

Matplotlib / Seaborn

Scikit-learn

Jupyter Notebook

🚀 Future Improvements

Hyperparameter tuning

Handle class imbalance (SMOTE)

Add explainability using SHAP

Deploy as a web app using Flask or Streamlit

📌 Conclusion

This project demonstrates how machine learning can assist in predicting heart disease using clinical data. By comparing multiple models and focusing on healthcare-relevant metrics, it highlights both the potential and limitations of ML in medical applications.

If you want next:

⭐ Resume bullet points

🎯 Interview questions from this project

🌐 Streamlit deployment README add-on
