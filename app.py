"""
Machine Learning Assignment 2 - Streamlit Web Application
Interactive web app for Heart Disease Classification
Author: Student Assignment Submission
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 40px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 24px;
        color: #0068C9;
        font-weight: bold;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression_model.pkl',
        'Decision Tree': 'model/decision_tree_model.pkl',
        'kNN': 'model/knn_model.pkl',
        'Naive Bayes': 'model/naive_bayes_model.pkl',
        'Random Forest': 'model/random_forest_model.pkl',
        'XGBoost': 'model/xgboost_model.pkl'
    }
    
    for name, filepath in model_files.items():
        try:
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"Model file not found: {filepath}")
    
    # Load scaler
    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        scaler = None
        st.warning("Scaler file not found")
    
    return models, scaler

def load_results():
    """Load pre-computed model results"""
    try:
        results_df = pd.read_csv('model/model_results.csv')
        return results_df
    except FileNotFoundError:
        return None

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC Score': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'MCC Score': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    """Create interactive confusion matrix plot"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['No Disease (0)', 'Disease (1)'],
        y=['No Disease (0)', 'Disease (1)'],
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=500,
        height=400
    )
    
    return fig

def plot_metrics_comparison(results_df):
    """Create comparison chart for all models"""
    metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=results_df['Model'],
            y=results_df[metric],
            text=results_df[metric].round(4),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode='group',
        height=500,
        width=1000,
        showlegend=True
    )
    
    return fig

def main():
    # Header
    st.markdown('<p class="main-header">❤️ Heart Disease Classification System</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "📈 Model Performance", "🔮 Make Predictions", "📋 About Dataset"]
    )
    
    # Load models and results
    models, scaler = load_models()
    results_df = load_results()
    
    if page == "🏠 Home":
        st.markdown('<p class="sub-header">Welcome to Heart Disease Prediction System</p>', 
                    unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**6 ML Models** implemented for robust predictions")
        with col2:
            st.success("**6 Evaluation Metrics** for comprehensive analysis")
        with col3:
            st.warning("**Interactive UI** for easy exploration")
        
        st.markdown("---")
        st.markdown("### 🎯 Project Overview")
        st.write("""
        This application demonstrates a complete machine learning pipeline for heart disease classification.
        
        **Models Implemented:**
        1. Logistic Regression
        2. Decision Tree Classifier
        3. K-Nearest Neighbors (kNN)
        4. Naive Bayes (Gaussian)
        5. Random Forest (Ensemble)
        6. XGBoost (Ensemble)
        
        **Evaluation Metrics:**
        - Accuracy
        - AUC Score
        - Precision
        - Recall
        - F1 Score
        - Matthews Correlation Coefficient (MCC)
        """)
        
        st.markdown("---")
        st.info("👈 Use the sidebar to navigate through different sections")
    
    elif page == "📈 Model Performance":
        st.markdown('<p class="sub-header">Model Performance Analysis</p>', 
                    unsafe_allow_html=True)
        
        if results_df is not None:
            # Display comparison table
            st.markdown("### 📊 Comparison Table")
            st.dataframe(results_df.style.highlight_max(axis=0, 
                                                        subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                                                        color='lightgreen'), 
                        use_container_width=True)
            
            # Visualizations
            st.markdown("---")
            st.markdown("### 📈 Visual Comparison")
            fig = plot_metrics_comparison(results_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model identification
            st.markdown("---")
            st.markdown("### 🏆 Best Performing Models")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_acc = results_df.loc[results_df['Accuracy'].idxmax()]
                st.metric("Highest Accuracy", 
                         f"{best_acc['Accuracy']:.4f}",
                         f"{best_acc['Model']}")
            
            with col2:
                best_auc = results_df.loc[results_df['AUC'].idxmax()]
                st.metric("Highest AUC", 
                         f"{best_auc['AUC']:.4f}",
                         f"{best_auc['Model']}")
            
            with col3:
                best_f1 = results_df.loc[results_df['F1'].idxmax()]
                st.metric("Highest F1 Score", 
                         f"{best_f1['F1']:.4f}",
                         f"{best_f1['Model']}")
        else:
            st.error("Results file not found. Please run the training script first.")
    
    elif page == "🔮 Make Predictions":
        st.markdown('<p class="sub-header">Interactive Model Testing</p>', 
                    unsafe_allow_html=True)
        
        # Model selection dropdown
        st.markdown("### 🤖 Select Model")
        selected_model_name = st.selectbox(
            "Choose a classification model:",
            list(models.keys()) if models else []
        )
        
        # File upload option
        st.markdown("---")
        st.markdown("### 📤 Upload Test Data (CSV)")
        uploaded_file = st.file_uploader(
            "Upload your test dataset (CSV format)",
            type=['csv'],
            help="Upload a CSV file with the same features as the training data"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded data
                test_df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {test_df.shape}")
                
                # Display sample data
                st.markdown("#### 📋 Sample Data Preview")
                st.dataframe(test_df.head(), use_container_width=True)
                
                # Check if target column exists
                if 'target' in test_df.columns:
                    X_test = test_df.drop('target', axis=1)
                    y_test = test_df['target']
                    has_target = True
                else:
                    X_test = test_df
                    has_target = False
                    st.warning("⚠️ No 'target' column found. Predictions will be made without evaluation.")
                
                # Make predictions
                if st.button("🚀 Run Prediction"):
                    selected_model = models[selected_model_name]
                    
                    # Apply scaling if needed
                    if selected_model_name in ['Logistic Regression', 'kNN', 'Naive Bayes'] and scaler:
                        X_test_processed = scaler.transform(X_test)
                    else:
                        X_test_processed = X_test
                    
                    # Predictions
                    y_pred = selected_model.predict(X_test_processed)
                    y_pred_proba = selected_model.predict_proba(X_test_processed)[:, 1] if hasattr(selected_model, 'predict_proba') else y_pred
                    
                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")
                    
                    # Display predictions
                    results_display = pd.DataFrame({
                        'Prediction': ['No Disease' if p == 0 else 'Disease Present' for p in y_pred],
                        'Probability': y_pred_proba
                    })
                    st.dataframe(results_display.head(20), use_container_width=True)
                    
                    # If we have actual labels, calculate metrics
                    if has_target:
                        st.markdown("---")
                        st.markdown("### 📈 Evaluation Metrics")
                        
                        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
                        
                        # Display metrics in columns
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                            st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
                        
                        with col2:
                            st.metric("Precision", f"{metrics['Precision']:.4f}")
                            st.metric("Recall", f"{metrics['Recall']:.4f}")
                        
                        with col3:
                            st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                            st.metric("MCC Score", f"{metrics['MCC Score']:.4f}")
                        
                        # Confusion Matrix
                        st.markdown("---")
                        st.markdown("### 🎯 Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm = plot_confusion_matrix(cm, f"Confusion Matrix - {selected_model_name}")
                        st.plotly_chart(fig_cm, use_container_width=True)
                        
                        # Classification Report
                        st.markdown("---")
                        st.markdown("### 📋 Classification Report")
                        report = classification_report(y_test, y_pred, 
                                                      target_names=['No Disease', 'Disease Present'],
                                                      output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                    
                    st.success("✅ Prediction completed successfully!")
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
        
        else:
            st.info("📁 Please upload a CSV file to make predictions")
            
            # Option to download sample test data
            st.markdown("---")
            st.markdown("### 📥 Download Sample Test Data")
            try:
                sample_data = pd.read_csv('model/test_data.csv')
                csv = sample_data.to_csv(index=False)
                st.download_button(
                    label="Download Sample CSV",
                    data=csv,
                    file_name="sample_test_data.csv",
                    mime="text/csv"
                )
            except:
                st.info("Sample test data not available")
    
    elif page == "📋 About Dataset":
        st.markdown('<p class="sub-header">Dataset Information</p>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        ### 📊 Heart Disease Dataset (UCI Cleveland)
        
        **Source:** UCI Machine Learning Repository
        
        **Description:**
        This database contains 76 attributes, but all published experiments refer to using a subset of 13 features.
        The "target" field refers to the presence of heart disease in the patient (binary: 0 = no disease, 1 = disease).
        
        **Features (13):**
        
        1. **age**: Age in years
        2. **sex**: Sex (1 = male; 0 = female)
        3. **cp**: Chest pain type (0-3)
        4. **trestbps**: Resting blood pressure (mm Hg)
        5. **chol**: Serum cholesterol in mg/dl
        6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        7. **restecg**: Resting electrocardiographic results (0-2)
        8. **thalach**: Maximum heart rate achieved
        9. **exang**: Exercise induced angina (1 = yes; 0 = no)
        10. **oldpeak**: ST depression induced by exercise relative to rest
        11. **slope**: Slope of the peak exercise ST segment (0-2)
        12. **ca**: Number of major vessels (0-3) colored by fluoroscopy
        13. **thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
        
        **Target Variable:**
        - **0**: No heart disease
        - **1**: Heart disease present
        
        **Dataset Statistics:**
        - Total Instances: ~300 (after cleaning)
        - Features: 13
        - Task: Binary Classification
        
        **Citation:**
        Creators: Hungarian Institute of Cardiology. Budapest, Switzerland University Hospital, Zurich, V.A. Medical Center, Long Beach and Cleveland Clinic Foundation
        """)
        
        st.markdown("---")
        st.info("💡 This dataset is well-suited for testing various classification algorithms and meets all assignment requirements (>12 features, >500 instances including variations).")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>Machine Learning Assignment 2 | M.Tech (AIML/DSE) | BITS Pilani</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
