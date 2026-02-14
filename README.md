# Machine Learning Assignment 2 - Heart Disease Classification

**ASWATHY H | 2025AB05203 | M.Tech (AIML/DSE)**

---

## Problem Statement

Develop a machine learning classification system to predict the presence of heart disease in patients based on clinical parameters. The system implements and compares six different classification algorithms and evaluates their performance using multiple metrics.

---

## Dataset Description

**Name**: Combined Heart Disease Dataset (UCI Repository)  
**Source**: UCI Machine Learning Repository  
**Data Sources**: Cleveland, Hungarian, Switzerland, Long Beach VA  
**URL**: https://archive.ics.uci.edu/ml/datasets/heart+disease  
**Total Instances**: 920  
**Number of Features**: 13  
**Target Variable**: Binary (0 = No disease, 1 = Disease present)  
**Task Type**: Binary Classification

### Features

| Feature | Description | Type | Range/Values |
|---------|-------------|------|--------------|
| **age** | Age of the patient in years | Continuous | 29-77 |
| **sex** | Gender of the patient | Categorical | 0 = Female, 1 = Male |
| **cp** | Chest pain type | Categorical | 0-3 (4 types) |
| **trestbps** | Resting blood pressure (mm Hg) | Continuous | 94-200 |
| **chol** | Serum cholesterol (mg/dl) | Continuous | 126-564 |
| **fbs** | Fasting blood sugar > 120 mg/dl | Binary | 0 = False, 1 = True |
| **restecg** | Resting electrocardiographic results | Categorical | 0-2 |
| **thalach** | Maximum heart rate achieved | Continuous | 71-202 |
| **exang** | Exercise induced angina | Binary | 0 = No, 1 = Yes |
| **oldpeak** | ST depression induced by exercise | Continuous | 0-6.2 |
| **slope** | Slope of peak exercise ST segment | Categorical | 0-2 |
| **ca** | Number of major vessels colored | Discrete | 0-3 |
| **thal** | Thalassemia | Categorical | 3, 6, 7 |
| **target** | Presence of heart disease (TARGET) | Binary | 0 = No, 1 = Yes |

---

## Models Used

### Comparison Table - Model Performance

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8261 | 0.8940 | 0.8431 | 0.8431 | 0.8431 | 0.6480 |
| Decision Tree | 0.8043 | 0.7929 | 0.7845 | 0.8922 | 0.8349 | 0.6047 |
| kNN | 0.8424 | 0.8677 | 0.8411 | 0.8824 | 0.8612 | 0.6801 |
| Naive Bayes | 0.8261 | 0.8875 | 0.8365 | 0.8529 | 0.8447 | 0.6473 |
| Random Forest (Ensemble) | 0.8315 | 0.9197 | 0.8381 | 0.8627 | 0.8502 | 0.6581 |
| XGBoost (Ensemble) | 0.8043 | 0.8786 | 0.8000 | 0.8627 | 0.8302 | 0.6026 |

---

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Solid baseline performance with balanced metrics across the board. Achieves consistent 0.8431 for Precision, Recall, and F1 Score. Good AUC (0.8940) indicates strong discriminative ability. Despite being a simple linear model, it provides interpretable coefficients and reliable predictions. Well-suited for this binary classification task. |
| **Decision Tree** | Good performance with highest recall (0.8922) among all models, making it effective at identifying disease cases. However, lower precision (0.7845) suggests more false positives. The model captures non-linear relationships well and is highly interpretable. Tree depth limitation helps control overfitting while maintaining good generalization. |
| **kNN** | Best overall accuracy (0.8424) and second-highest F1 score (0.8612). Distance-based approach performs well after proper feature scaling. High recall (0.8824) makes it suitable for medical diagnosis. Strong MCC (0.6801) indicates reliable predictions. Computationally efficient with proper preprocessing. |
| **Naive Bayes** | Strong probabilistic performance with excellent AUC (0.8875) and balanced metrics. Works remarkably well despite feature independence assumption. Very fast training and prediction, ideal for real-time applications. Consistent performance (Accuracy: 0.8261) demonstrates robustness. Good probability calibration for decision-making. |
| **Random Forest (Ensemble)** | Highest AUC score (0.9197) indicating best discriminative ability. Solid ensemble performance with excellent generalization through bagging. Balanced metrics and robust to outliers. Provides valuable feature importance insights. Handles mixed data types effectively. Strong overall performance (Accuracy: 0.8315, F1: 0.8502). |
| **XGBoost (Ensemble)** | Competitive ensemble performance with high recall (0.8627). Gradient boosting provides good sequential error correction. Balanced precision-recall trade-off. While not the best in individual metrics, maintains consistent performance across all measures. Suitable for production deployment with good reliability. |

---

**ASWATHY H | 2025AB05203 | BITS Pilani**
