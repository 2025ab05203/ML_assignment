"""
Heart Disease Classification - Streamlit Web Application
Interactive web app for Heart Disease Classification
ASWATHY H | BITS ID: 2025AB05203
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
    page_title="Heart Disease ML Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&display=swap');
    
    * {
        font-family: monospace;
    }
    
    .main-header {
        font-size: 43px !important;
        font-weight: 900;
        text-align: center;
        padding: 10px 0 20px 0;
        margin-top: 0;
        margin-bottom: -13px !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        text-shadow: 3px 3px 6px rgba(220, 38, 38, 0.2);
        line-height: 1.1;
    }
    
    .sub-header {
        font-size: 28px;
        color: #1e293b;
        font-weight: 600;
        margin: 30px 0 20px 0;
        padding-bottom: 12px;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #1e40af, #3b82f6) 1;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        box-shadow: 2px 0 15px rgba(0,0,0,0.1);
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Sidebar Title */
    section[data-testid="stSidebar"] h1 {
        color: #ffffff !important;
        font-size: 24px !important;
        font-weight: 600 !important;
        padding: 20px 0 10px 0 !important;
        border-bottom: 2px solid rgba(255,255,255,0.1);
        margin-bottom: 20px !important;
    }
    
    /* Sidebar Navigation Buttons */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: rgba(255, 255, 255, 0.05);
        color: #e2e8f0 !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 14px 18px;
        font-size: 15px;
        font-weight: 500;
        text-align: left;
        margin: 6px 0;
        transition: all 0.3s ease;
        box-shadow: none;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(99, 102, 241, 0.4);
        transform: translateX(5px);
        box-shadow: 0 4px 8px rgba(99, 102, 241, 0.2);
    }
    
    section[data-testid="stSidebar"] .stButton > button:active {
        background: rgba(99, 102, 241, 0.35);
    }
    
    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }
    
    /* Metrics */
    .stMetric {
        background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border: 1px solid rgba(99, 102, 241, 0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white !important;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 14px 28px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(99, 102, 241, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    }
    
    h1, h2, h3 {
        color: #1e293b;
    }
    
    /* DataFrames */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Hide only the toolbar/deploy button area */
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    [data-testid="stDecoration"] {
        display: none !important;
    }
    
    /* Keep header visible but minimal */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Make sure sidebar is always visible even when collapsed */
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
    }
    
    /* When sidebar is collapsed, keep it partially visible */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: translateX(0) !important;
        min-width: 80px !important;
        width: 80px !important;
        overflow: visible !important;
    }
    
    /* Hide navigation heading and text when sidebar is collapsed */
    section[data-testid="stSidebar"][aria-expanded="false"] h1,
    section[data-testid="stSidebar"][aria-expanded="false"] p,
    section[data-testid="stSidebar"][aria-expanded="false"] .sidebar-nav-header {
        display: none !important;
    }
    
    /* Hide navigation buttons when sidebar is collapsed (but not the toggle button) */
    section[data-testid="stSidebar"][aria-expanded="false"] .stButton {
        display: none !important;
    }
    
    /* Streamlit's native collapse/expand button styling */
    button[kind="header"] {
        background: #DC2626 !important;
        color: white !important;
        border: none !important;
        padding: 8px 12px !important;
        cursor: pointer !important;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.3) !important;
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="header"]:hover {
        background: #B91C1C !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.5) !important;
    }
    
    /* When sidebar is expanded, button is at top-right of sidebar */
    section[data-testid="stSidebar"][aria-expanded="true"] button[kind="header"] {
        border-radius: 8px !important;
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* When sidebar is collapsed, button appears at right edge of collapsed sidebar */
    section[data-testid="stSidebar"][aria-expanded="false"] button[kind="header"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        left: 20px !important;
        top: 16px !important;
        z-index: 999999 !important;
        background: #DC2626 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
        box-shadow: 0 2px 12px rgba(220, 38, 38, 0.4) !important;
        border: none !important;
        cursor: pointer !important;
        min-width: 40px !important;
        min-height: 40px !important;
        pointer-events: auto !important;
    }
    
    /* Absolutely ensure button is not hidden by any other rule - always visible */
    section[data-testid="stSidebar"][aria-expanded="false"] button[kind="header"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    [data-testid="collapsedControl"]:hover,
    section[data-testid="stSidebar"][aria-expanded="false"] button[kind="header"]:hover {
        background: #B91C1C !important;
        box-shadow: 0 4px 16px rgba(220, 38, 38, 0.6) !important;
    }
    
    /* Make sure the icon/svg is visible and properly sized */
    button[kind="header"] svg,
    [data-testid="collapsedControl"] svg {
        color: white !important;
        fill: white !important;
        width: 20px !important;
        height: 20px !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* When sidebar is collapsed, hide the arrow SVG completely */
    section[data-testid="stSidebar"][aria-expanded="false"] button[kind="header"] svg,
    [data-testid="collapsedControl"] svg {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
    }
    
    /* Hide the span containing the icon text when collapsed */
    section[data-testid="stSidebar"][aria-expanded="false"] button[kind="header"] span,
    [data-testid="collapsedControl"] span {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Show menu icon (three horizontal lines) when collapsed - using Material Icons font */
    section[data-testid="stSidebar"][aria-expanded="false"] button[kind="header"]::after,
    [data-testid="collapsedControl"]::after {
        content: "menu" !important; /* here-is-the-icon - Material Icons: "menu" for hamburger icon */
        font-family: 'Material Icons' !important;
        font-size: 24px !important;
        color: white !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 1 !important;
        font-weight: normal !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
    }
    
    /* Keep arrow pointing left when sidebar is expanded */
    section[data-testid="stSidebar"][aria-expanded="true"] button[kind="header"] svg {
        transform: scaleX(1) !important;
        transition: transform 0.3s ease !important;
    }
    
    /* Force the button to always be visible, especially when collapsed */
    section[data-testid="stSidebar"] button[kind="header"],
    section[data-testid="stSidebar"] [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
    }
    
    /* Override any parent styles that might hide the button */
    button[kind="header"],
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* Ensure main content area allows space for toggle button */
    .main .block-container {
        padding-left: 2rem !important;
    }
    
    /* Custom Loading Animation */
    .stSpinner > div {
        border-top-color: #1e40af !important;
        animation: spinner-border 0.75s linear infinite !important;
    }
    
            .st-emotion-cache-qmp9ai{
            visibility: visible !important;
            }
    @keyframes spinner-border {
        to { transform: rotate(360deg); }
    }
    </style>
    
    <script>
    // Force sidebar to be visible
    window.addEventListener('load', function() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.setAttribute('aria-expanded', 'true');
            sidebar.style.transform = 'translateX(0)';
            sidebar.style.display = 'block';
            sidebar.style.visibility = 'visible';
        }
    });
    </script>
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
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Header
    st.markdown('<p class="main-header">🫀 Heart Disease ML Classifier</p>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar with custom header
    with st.sidebar:
        st.markdown("""
            <div class='sidebar-nav-header' style='text-align: center; padding: 20px 0; margin-bottom: -38px;'>
                <h1 style='font-size: 28px; margin: 0; color: white !important;'>Navigation</h1>
            </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons
        if st.button("Home", use_container_width=True, key="nav_home"):
            st.session_state.current_page = "Home"
            st.rerun()
        
        if st.button("Train Models", use_container_width=True, key="nav_train"):
            st.session_state.current_page = "Train Models"
            st.rerun()
        
        if st.button("Model Performance", use_container_width=True, key="nav_performance"):
            st.session_state.current_page = "Model Performance"
            st.rerun()
        
        if st.button("Make Predictions", use_container_width=True, key="nav_predict"):
            st.session_state.current_page = "Make Predictions"
            st.rerun()
        
        if st.button("About Dataset", use_container_width=True, key="nav_dataset"):
            st.session_state.current_page = "About Dataset"
            st.rerun()
        
        # Sidebar footer
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; padding: 10px; color: #94a3b8; font-size: 12px;'>
                <p style='margin: 5px 0;'>ASWATHY H | BITS ID: 2025AB05203</p>
                <p style='margin: 5px 0;'>M.Tech AIML/DSE</p>
                <p style='margin: 5px 0;'>BITS Pilani</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Get current page
    page = st.session_state.current_page
    
    # Load models and results
    models, scaler = load_models()
    results_df = load_results()
    
    if page == "Home":
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
        st.markdown("### Project Overview")
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
    
    elif page == "Train Models":
        st.markdown('<p class="sub-header">Train Classification Models</p>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        ### Training All 6 Machine Learning Models
        
        Training the following models on the Heart Disease dataset:
        1. Logistic Regression
        2. Decision Tree Classifier
        3. K-Nearest Neighbors (kNN)
        4. Naive Bayes (Gaussian)
        5. Random Forest (Ensemble)
        6. XGBoost (Ensemble)
        """)
        
        st.markdown("---")
        
        if True:
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.naive_bayes import GaussianNB
                from sklearn.ensemble import RandomForestClassifier
                from xgboost import XGBClassifier
                
                status_text.text("Loading combined datasets...")
                progress_bar.progress(10)
                
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
                        df_temp = pd.read_csv(url, names=column_names, na_values='?')
                        all_dataframes.append(df_temp)
                    except:
                        pass
                
                df_combined = pd.concat(all_dataframes, ignore_index=True)
                
                X_temp = df_combined.drop('target', axis=1)
                y_temp = df_combined['target']
                
                mask = y_temp.notna()
                X_temp = X_temp[mask]
                y_temp = y_temp[mask]
                
                imputer = SimpleImputer(strategy='median')
                X_imputed = pd.DataFrame(
                    imputer.fit_transform(X_temp),
                    columns=X_temp.columns
                )
                
                y_temp = (y_temp > 0).astype(int)
                df = pd.concat([X_imputed, y_temp.reset_index(drop=True)], axis=1)
                
                st.success(f"✅ Combined dataset loaded: {df.shape[0]} instances, {df.shape[1]-1} features")
                st.info(f"📊 Data sources: Cleveland, Hungarian, Switzerland, Long Beach VA (Missing values imputed using median strategy)")
                progress_bar.progress(20)
                
                status_text.text("Preparing data...")
                X = df.drop('target', axis=1)
                y = df['target']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                with open('model/scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                
                progress_bar.progress(30)
                
                models_to_train = {
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
                    'kNN': KNeighborsClassifier(n_neighbors=5),
                    'Naive Bayes': GaussianNB(),
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
                }
                
                results = []
                progress_step = 60 / len(models_to_train)
                current_progress = 30
                
                for idx, (name, model) in enumerate(models_to_train.items(), 1):
                    status_text.text(f"Training {name} ({idx}/{len(models_to_train)})...")
                    
                    if name in ['Logistic Regression', 'kNN', 'Naive Bayes']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                    
                    metrics = {
                        'Model': name,
                        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
                        'AUC': round(roc_auc_score(y_test, y_pred_proba), 4),
                        'Precision': round(precision_score(y_test, y_pred, average='binary', zero_division=0), 4),
                        'Recall': round(recall_score(y_test, y_pred, average='binary', zero_division=0), 4),
                        'F1': round(f1_score(y_test, y_pred, average='binary', zero_division=0), 4),
                        'MCC': round(matthews_corrcoef(y_test, y_pred), 4)
                    }
                    results.append(metrics)
                    
                    filename = f"model/{name.replace(' ', '_').lower()}_model.pkl"
                    with open(filename, 'wb') as f:
                        pickle.dump(model, f)
                    
                    current_progress += progress_step
                    progress_bar.progress(int(current_progress))
                
                status_text.text("Saving results...")
                progress_bar.progress(95)
                
                results_df = pd.DataFrame(results)
                results_df.to_csv('model/model_results.csv', index=False)
                
                test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
                test_data.to_csv('model/test_data.csv', index=False)
                
                progress_bar.progress(100)
                status_text.text("Training completed!")
                
                st.success("All models trained successfully!")
                
                st.markdown("---")
                st.markdown("### Training Results")
                st.dataframe(results_df.style.highlight_max(axis=0, 
                                                           subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                                                           color='lightgreen'),
                            use_container_width=True)
                
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    best_acc = results_df.loc[results_df['Accuracy'].idxmax()]
                    st.metric("Best Accuracy", 
                             f"{best_acc['Accuracy']:.4f}",
                             f"{best_acc['Model']}")
                
                with col2:
                    best_auc = results_df.loc[results_df['AUC'].idxmax()]
                    st.metric("Best AUC", 
                             f"{best_auc['AUC']:.4f}",
                             f"{best_auc['Model']}")
                
                with col3:
                    best_f1 = results_df.loc[results_df['F1'].idxmax()]
                    st.metric("Best F1 Score", 
                             f"{best_f1['F1']:.4f}",
                             f"{best_f1['Model']}")
                
                # Clear cache to reload models
                st.cache_resource.clear()
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")
                status_text.text("Error occurred during training")
                progress_bar.progress(0)
    
    elif page == "Model Performance":
        st.markdown('<p class="sub-header">Model Performance Analysis</p>', 
                    unsafe_allow_html=True)
        
        if results_df is not None and len(results_df) > 0:
            # Display comparison table
            st.markdown("### Comparison Table")
            st.dataframe(results_df.style.highlight_max(axis=0, 
                                                        subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                                                        color='lightgreen'), 
                        use_container_width=True)
            
            # Visualizations
            st.markdown("---")
            st.markdown("### Visual Comparison")
            fig = plot_metrics_comparison(results_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Best model identification
            st.markdown("---")
            st.markdown("### Best Performing Models")
            
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
            st.warning("No training results found!")
            st.info("Please go to **Train Models** page and train the models first.")
            st.markdown("""
            ### Quick Start:
            1. Click on **Train Models** in the sidebar
            2. Click the **Start Training** button
            3. Wait for training to complete (~30-60 seconds)
            4. Return here to view performance metrics
            """)
    
    elif page == "Make Predictions":
        st.markdown('<p class="sub-header">Interactive Model Testing</p>', 
                    unsafe_allow_html=True)
        
        # Model selection dropdown
        st.markdown("### Select Model")
        selected_model_name = st.selectbox(
            "Choose a classification model:",
            list(models.keys()) if models else []
        )
        
        # File upload option
        st.markdown("---")
        st.markdown("### Upload Test Data (CSV)")
        uploaded_file = st.file_uploader(
            "Upload your test dataset (CSV format)",
            type=['csv'],
            help="Upload a CSV file with the same features as the training data"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded data
                test_df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Shape: {test_df.shape}")
                
                # Display sample data
                st.markdown("#### Sample Data Preview")
                st.dataframe(test_df.head(), use_container_width=True)
                
                # Check if target column exists
                if 'target' in test_df.columns:
                    X_test = test_df.drop('target', axis=1)
                    y_test = test_df['target']
                    has_target = True
                else:
                    X_test = test_df
                    has_target = False
                    st.warning("No 'target' column found. Predictions will be made without evaluation.")
                
                # Make predictions
                if st.button("Run Prediction"):
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
                    st.markdown("### Prediction Results")
                    
                    # Display predictions
                    results_display = pd.DataFrame({
                        'Prediction': ['No Disease' if p == 0 else 'Disease Present' for p in y_pred],
                        'Probability': y_pred_proba
                    })
                    st.dataframe(results_display.head(20), use_container_width=True)
                    
                    # If we have actual labels, calculate metrics
                    if has_target:
                        st.markdown("---")
                        st.markdown("### Evaluation Metrics")
                        
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
                        st.markdown("### Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm = plot_confusion_matrix(cm, f"Confusion Matrix - {selected_model_name}")
                        st.plotly_chart(fig_cm, use_container_width=True)
                        
                        # Classification Report
                        st.markdown("---")
                        st.markdown("### Classification Report")
                        report = classification_report(y_test, y_pred, 
                                                      target_names=['No Disease', 'Disease Present'],
                                                      output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                    
                    st.success("Prediction completed successfully!")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        else:
            st.info("Please upload a CSV file to make predictions")
            
            # Option to download sample test data
            st.markdown("---")
            st.markdown("### Download Sample Test Data")
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
    
    elif page == "About Dataset":
        st.markdown('<p class="sub-header">Dataset Information</p>', 
                    unsafe_allow_html=True)
        
        st.markdown("""
        ### Combined Heart Disease Dataset (UCI Repository)
        
        **Source:** UCI Machine Learning Repository
        
        **Description:**
        This implementation uses a combined dataset from 4 different medical institutions to ensure robust model training with sufficient data:
        
        **Data Sources:**
        1. **Cleveland Clinic Foundation** - Ohio, USA
        2. **Hungarian Institute of Cardiology** - Budapest, Hungary
        3. **V.A. Medical Center** - Long Beach, California, USA
        4. **University Hospital** - Zurich, Switzerland
        
        Each database contains 76 attributes, but all published experiments refer to using a subset of 13 features.
        The "target" field refers to the presence of heart disease in the patient (binary: 0 = no disease, 1-4 = disease present).
        
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
        - **1**: Heart disease present (original values 1-4 converted to binary)
        
        **Dataset Statistics:**
        - **Total Instances**: 920 (combined from all 4 sources)
        - **After Removing Invalid Target Values**: ~900 instances
        - **Features**: 13
        - **Task**: Binary Classification
        - **Missing Data Handling**: Median Imputation (better ML practice than deletion)
        
        **Data Preprocessing:**
        - Missing values in features are imputed using median strategy
        - Rows with missing target values are removed
        - Target values 1-4 are converted to binary (1 = disease present)
        - All 4 datasets preserve full 13-feature structure
        
        **Citation:**
        Creators: Hungarian Institute of Cardiology, Budapest; University Hospital, Zurich, Switzerland; V.A. Medical Center, Long Beach, California; Cleveland Clinic Foundation, Ohio, USA
        
        **Reference:**
        Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64, 304-310.
        """)
        
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 30px 20px; background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 100%); border-radius: 10px; margin-top: 50px;'>
            <p style='color: #475569; font-size: 14px; margin: 5px 0; font-weight: 500;'>ASWATHY H | BITS ID: 2025AB05203</p>
            <p style='color: #64748b; font-size: 13px; margin: 5px 0;'>M.Tech (AIML/DSE) • BITS Pilani</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
