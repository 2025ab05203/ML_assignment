"""
Machine Learning Assignment 2 - Model Training and Evaluation
Implements 6 classification models with comprehensive evaluation metrics
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load Combined Heart Disease dataset from 4 UCI repositories
    Datasets: Cleveland, Hungarian, Switzerland, Long Beach VA
    Features: 13 attributes
    Target: Binary classification (0 = no disease, 1 = disease)
    Combined Total: 900+ instances (using imputation for missing values)
    """
    from sklearn.impute import SimpleImputer
    
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    datasets_urls = {
        'Cleveland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
        'Hungarian': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data',
        'Switzerland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data',
        'Long Beach VA': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data'
    }
    
    all_dataframes = []
    
    for name, url in datasets_urls.items():
        try:
            df = pd.read_csv(url, names=column_names, na_values='?')
            print(f"Loaded {name}: {df.shape[0]} instances")
            all_dataframes.append(df)
        except Exception as e:
            print(f"Warning: Could not load {name} dataset: {e}")
    
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nCombined dataset before cleaning: {df_combined.shape[0]} instances")
    
    X = df_combined.drop('target', axis=1)
    y = df_combined['target']
    
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )
    
    y_binary = (y > 0).astype(int)
    
    df_final = pd.concat([X_imputed, y_binary.reset_index(drop=True)], axis=1)
    
    print(f"\n{'='*60}")
    print(f"COMBINED HEART DISEASE DATASET")
    print(f"{'='*60}")
    print(f"Total Instances: {df_final.shape[0]}")
    print(f"Features: {df_final.shape[1] - 1}")
    print(f"Task: Binary Classification")
    print(f"Missing Value Strategy: Median Imputation")
    print(f"\nClass distribution:\n{df_final['target'].value_counts()}")
    print(f"{'='*60}\n")
    
    return df_final

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train model and calculate all 6 required evaluation metrics
    """
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics = {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(roc_auc_score(y_test, y_pred_proba), 4),
        'Precision': round(precision_score(y_test, y_pred, average='binary', zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, average='binary', zero_division=0), 4),
        'F1': round(f1_score(y_test, y_pred, average='binary', zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        if metric != 'Model':
            print(f"{metric:15s}: {value}")
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics, model

def main():
    """
    Main function to train all 6 models and save results
    """
    print("="*60)
    print("ML Assignment 2 - Heart Disease Classification")
    print("="*60)
    
    df = load_and_prepare_data()
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        if name in ['Logistic Regression', 'kNN', 'Naive Bayes']:
            metrics, trained_model = evaluate_model(
                model, X_train_scaled, X_test_scaled, y_train, y_test, name
            )
        else:
            metrics, trained_model = evaluate_model(
                model, X_train, X_test, y_train, y_test, name
            )
        
        results.append(metrics)
        trained_models[name] = trained_model
    
    results_df = pd.DataFrame(results)
    results_df = results_df[['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
    
    print("\n" + "="*80)
    print("COMPARISON TABLE - ALL MODELS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    results_df.to_csv('model_results.csv', index=False)
    print("\nResults saved to 'model_results.csv'")
    
    for name, model in trained_models.items():
        filename = f"{name.replace(' ', '_').lower()}_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved: {filename}")
    
    test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    test_data.to_csv('test_data.csv', index=False)
    print("\nTest data saved to 'test_data.csv'")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    
    return results_df, trained_models

if __name__ == "__main__":
    results_df, models = main()
