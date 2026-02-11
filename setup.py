"""
Quick Setup and Verification Script
Machine Learning Assignment 2
"""

import subprocess
import sys
import os

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_python_version():
    """Verify Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.8 or higher required")
        return False

def install_dependencies():
    """Install required packages"""
    print_header("Installing Dependencies")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False

def verify_file_structure():
    """Verify all required files exist"""
    print_header("Verifying Project Structure")
    
    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        "model/ml_models_training.py"
    ]
    
    all_present = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            all_present = False
    
    return all_present

def train_models():
    """Run model training script"""
    print_header("Training Models")
    print("This will take a few minutes...")
    
    try:
        os.chdir("model")
        subprocess.check_call([sys.executable, "ml_models_training.py"])
        os.chdir("..")
        print("\n✓ Model training completed successfully")
        return True
    except subprocess.CalledProcessError:
        print("\n✗ Model training failed")
        os.chdir("..")
        return False

def verify_model_files():
    """Check if model files were created"""
    print_header("Verifying Model Files")
    
    model_files = [
        "model/logistic_regression_model.pkl",
        "model/decision_tree_model.pkl",
        "model/knn_model.pkl",
        "model/naive_bayes_model.pkl",
        "model/random_forest_model.pkl",
        "model/xgboost_model.pkl",
        "model/scaler.pkl",
        "model/model_results.csv",
        "model/test_data.csv"
    ]
    
    all_present = True
    for file in model_files:
        if os.path.exists(file):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} - MISSING")
            all_present = False
    
    return all_present

def main():
    """Main setup function"""
    print("\n" + "╔" + "="*58 + "╗")
    print("║  ML Assignment 2 - Automated Setup & Verification      ║")
    print("╚" + "="*58 + "╝")
    
    # Step 1: Check Python version
    if not check_python_version():
        print("\n⚠️  Please install Python 3.8 or higher")
        return
    
    # Step 2: Verify file structure
    if not verify_file_structure():
        print("\n⚠️  Some required files are missing!")
        return
    
    # Step 3: Install dependencies
    response = input("\nInstall dependencies? (y/n): ")
    if response.lower() == 'y':
        if not install_dependencies():
            print("\n⚠️  Dependency installation failed")
            return
    
    # Step 4: Train models
    response = input("\nTrain models now? This will take a few minutes. (y/n): ")
    if response.lower() == 'y':
        if train_models():
            verify_model_files()
        else:
            print("\n⚠️  Model training failed. Check errors above.")
            return
    
    # Step 5: Launch Streamlit
    print_header("Setup Complete!")
    print("✓ All checks passed")
    print("\nNext steps:")
    print("1. Run 'streamlit run app.py' to start the web application")
    print("2. Upload to GitHub")
    print("3. Deploy to Streamlit Community Cloud")
    print("4. Submit assignment on Taxila")
    
    response = input("\nLaunch Streamlit app now? (y/n): ")
    if response.lower() == 'y':
        print("\nStarting Streamlit app...")
        print("The app will open in your browser at http://localhost:8501")
        print("Press Ctrl+C to stop the server\n")
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        except KeyboardInterrupt:
            print("\n\nStreamlit server stopped.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
    except Exception as e:
        print(f"\n\n⚠️  An error occurred: {str(e)}")
