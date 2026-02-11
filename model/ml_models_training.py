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
    Load Heart Disease dataset and prepare for training
    Dataset: Heart Disease UCI (Cleveland)
    Features: 13 attributes
    Target: Binary classification (0 = no disease, 1 = disease)
    """
    # Load dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    df = pd.read_csv(url, names=column_names, na_values='?')
    
    # Handle missing values
    df = df.dropna()
    
    # Convert target to binary (0: no disease, 1-4: disease present)
    df['target'] = (df['target'] > 0).astype(int)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1}")
    print(f"Instances: {df.shape[0]}")
    print(f"\nClass distribution:\n{df['target'].value_counts()}")
    
    return df

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train model and calculate all 6 required evaluation metrics
    """
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'AUC': round(roc_auc_score(y_test, y_pred_proba), 4),
        'Precision': round(precision_score(y_test, y_pred, average='binary', zero_division=0), 4),
        'Recall': round(recall_score(y_test, y_pred, average='binary', zero_division=0), 4),
        'F1': round(f1_score(y_test, y_pred, average='binary', zero_division=0), 4),
        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
    }
    
    # Print detailed results
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
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature scaling (important for some models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for later use
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize all 6 models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    }
    
    # Train and evaluate each model
    results = []
    trained_models = {}
    
    for name, model in models.items():
        # Use scaled data for models that benefit from it
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
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df[['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
    
    # Display comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE - ALL MODELS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    # Save results to CSV
    results_df.to_csv('model_results.csv', index=False)
    print("\nResults saved to 'model_results.csv'")
    
    # Save all trained models
    for name, model in trained_models.items():
        filename = f"{name.replace(' ', '_').lower()}_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved: {filename}")
    
    # Save test data for Streamlit app
    test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    test_data.to_csv('test_data.csv', index=False)
    print("\nTest data saved to 'test_data.csv'")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    
    return results_df, trained_models

if __name__ == "__main__":
    results_df, models = main()
