# 🎯 Step-by-Step Execution Guide
# Machine Learning Assignment 2

## 📋 COMPLETE CHECKLIST FOR FULL MARKS (15/15)

### ✅ Phase 1: Local Development and Testing

#### Step 1: Train the Models (10 marks)
```bash
# Navigate to model directory
cd model

# Run training script
python ml_models_training.py
```

**Expected Output:**
- Training progress for all 6 models
- Evaluation metrics for each model
- Comparison table printed
- Files created:
  ✓ logistic_regression_model.pkl
  ✓ decision_tree_model.pkl
  ✓ knn_model.pkl
  ✓ naive_bayes_model.pkl
  ✓ random_forest_model.pkl
  ✓ xgboost_model.pkl
  ✓ scaler.pkl
  ✓ model_results.csv
  ✓ test_data.csv

#### Step 2: Test Streamlit App Locally (4 marks)
```bash
# Navigate back to main directory
cd ..

# Run Streamlit app
streamlit run app.py
```

**Test All Features:**
1. ✓ Home page loads correctly
2. ✓ Model Performance page shows comparison table
3. ✓ Model Performance page shows visualization
4. ✓ Make Predictions - Model selection dropdown works
5. ✓ Make Predictions - File upload accepts CSV
6. ✓ Make Predictions - Displays evaluation metrics
7. ✓ Make Predictions - Shows confusion matrix
8. ✓ Make Predictions - Shows classification report
9. ✓ About Dataset page displays correctly

**IMPORTANT**: Test upload with model/test_data.csv to verify functionality!

---

### ✅ Phase 2: BITS Virtual Lab Execution (1 mark)

#### Step 3: Execute on BITS Virtual Lab
1. **Login to BITS Virtual Lab**
   - URL: [BITS Virtual Lab URL]
   - Use your credentials

2. **Upload Project Files**
   - Upload entire project folder OR
   - Clone from GitHub (if already pushed)

3. **Run Training Script**
   ```bash
   cd Assignment_2/model
   python ml_models_training.py
   ```

4. **Take Screenshot**
   - Screenshot must show:
     ✓ BITS Virtual Lab interface
     ✓ Your terminal with training output
     ✓ Successful model training completion
     ✓ Timestamp/date visible
   
5. **Save Screenshot**
   - Name: `BITS_Lab_Execution.png` or similar
   - Keep ready for PDF submission

---

### ✅ Phase 3: GitHub Repository Setup

#### Step 4: Create GitHub Repository

1. **Create New Repository**
   - Go to: https://github.com/new
   - Repository name: `ML-Assignment-2-Heart-Disease` (or similar)
   - Set as Public (required for Streamlit deployment)
   - DO NOT initialize with README (we have our own)

2. **Initialize Local Git**
   ```bash
   # In your project directory
   git init
   git add .
   git commit -m "Initial commit: ML Assignment 2 - Heart Disease Classification"
   ```

3. **Connect to GitHub**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

4. **Verify Upload**
   - Check GitHub repository has:
     ✓ app.py
     ✓ requirements.txt
     ✓ README.md
     ✓ .gitignore
     ✓ model/ folder with all .pkl files
     ✓ model/ml_models_training.py
     ✓ model/model_results.csv
     ✓ model/test_data.csv

---

### ✅ Phase 4: Streamlit Cloud Deployment

#### Step 5: Deploy to Streamlit Community Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://share.streamlit.io/
   - OR: https://streamlit.io/cloud

2. **Sign In with GitHub**
   - Click "Sign in"
   - Authorize with your GitHub account

3. **Deploy New App**
   - Click "New app" button
   - **Repository**: Select your GitHub repo
   - **Branch**: Select `main`
   - **Main file path**: `app.py`
   - **App URL** (optional): Choose custom name

4. **Advanced Settings** (if needed)
   - Python version: 3.9 or 3.10
   - Leave other settings as default

5. **Click "Deploy!"**
   - Wait 2-5 minutes for deployment
   - Monitor logs for any errors

6. **Deployment Success!**
   - You'll get URL like: `https://your-app-name.streamlit.app`
   - **COPY THIS URL** for submission

7. **Test Live App**
   - Visit the URL
   - Test all features:
     ✓ Navigation works
     ✓ Model Performance page loads
     ✓ File upload works
     ✓ Predictions generate correctly

---

### ✅ Phase 5: Submission Preparation

#### Step 6: Create Submission PDF

**PDF Must Include (IN THIS ORDER):**

1. **Cover Page**
   - Your Name
   - Course: M.Tech (AIML/DSE)
   - Subject: Machine Learning - Assignment 2
   - Date of Submission

2. **GitHub Repository Link**
   ```
   GitHub Repository: https://github.com/YOUR_USERNAME/YOUR_REPO_NAME
   ```
   - Make sure it's a clickable hyperlink!

3. **Live Streamlit App Link**
   ```
   Streamlit App: https://your-app-name.streamlit.app
   ```
   - Make sure it's a clickable hyperlink!

4. **BITS Virtual Lab Screenshot**
   - Insert the screenshot you took in Step 3
   - Should clearly show execution on BITS Lab

5. **Complete README Content**
   - Copy entire README.md content
   - Include:
     ✓ Problem Statement
     ✓ Dataset Description
     ✓ Comparison Table with all metrics
     ✓ Model Observations
     ✓ All sections from README

6. **Optional: Additional Screenshots**
   - Streamlit app screenshots
   - Model performance visualizations

**Save as**: `Assignment2_YourName.pdf`

---

### ✅ Phase 6: Final Submission

#### Step 7: Submit on Taxila

1. **Login to Taxila**
   - Go to BITS Taxila portal
   - Navigate to Machine Learning course

2. **Find Assignment 2**
   - Go to Assignments section
   - Click on "Assignment 2"

3. **Upload PDF**
   - Upload your prepared PDF
   - File size should be < 10 MB

4. **IMPORTANT: SUBMIT (Not Save Draft)**
   - Click "SUBMIT" button
   - Confirm submission
   - **NO RESUBMISSIONS ALLOWED**

5. **Verify Submission**
   - Check confirmation email/message
   - Verify submission timestamp

---

## 🎯 MARKS BREAKDOWN (Total: 15)

| Component | Marks | Verification |
|-----------|-------|--------------|
| Logistic Regression + Metrics | 1 | ✓ All 6 metrics calculated |
| Decision Tree + Metrics | 1 | ✓ All 6 metrics calculated |
| kNN + Metrics | 1 | ✓ All 6 metrics calculated |
| Naive Bayes + Metrics | 1 | ✓ All 6 metrics calculated |
| Random Forest + Metrics | 1 | ✓ All 6 metrics calculated |
| XGBoost + Metrics | 1 | ✓ All 6 metrics calculated |
| Dataset Description | 1 | ✓ Complete description in README |
| Model Observations | 3 | ✓ Detailed observations for all models |
| CSV Upload Feature | 1 | ✓ Working in Streamlit app |
| Model Selection Dropdown | 1 | ✓ All 6 models selectable |
| Metrics Display | 1 | ✓ All metrics shown |
| Confusion Matrix/Report | 1 | ✓ Both displayed |
| BITS Lab Screenshot | 1 | ✓ Included in PDF |
| **TOTAL** | **15** | |

---

## ⚠️ COMMON PITFALLS TO AVOID

❌ **DON'T:**
- Skip training models before deploying
- Forget to push model .pkl files to GitHub
- Use private GitHub repository
- Submit without testing live app
- Click "Save Draft" instead of "Submit"
- Copy-paste from other students
- Use identical variable names as templates

✅ **DO:**
- Test everything locally first
- Verify all links work before submission
- Add unique comments and observations
- Customize variable names
- Test file upload with actual CSV
- Double-check all requirements met
- Submit well before deadline

---

## 🐛 TROUBLESHOOTING

### Issue: Models not loading in Streamlit
**Solution**: Ensure all .pkl files are in model/ folder and pushed to GitHub

### Issue: Module not found error
**Solution**: Check requirements.txt has all dependencies with correct versions

### Issue: Streamlit app crashes on upload
**Solution**: Verify CSV has correct column names matching training data

### Issue: GitHub repository empty
**Solution**: Make sure you pushed: `git add . && git commit -m "update" && git push`

### Issue: Can't deploy on Streamlit Cloud
**Solution**: 
- Ensure repo is public
- Verify requirements.txt exists
- Check app.py is in root directory

---

## 📞 SUPPORT

**For BITS Lab Issues:**
- Email: neha.vinayak@pilani.bits-pilani.ac.in
- Subject: "ML Assignment 2: BITS Lab issue"

**For Technical Issues:**
- Check Streamlit documentation
- Verify GitHub repository structure
- Test locally before deploying

---

## ⏰ DEADLINE

**Submission Deadline**: 15-Feb-2026, 11:59 PM

**Time Management:**
- Day 1-2: Train models, test locally
- Day 3: Execute on BITS Lab, take screenshot
- Day 4: Setup GitHub, deploy to Streamlit
- Day 5: Create PDF, verify everything, SUBMIT

---

## ✨ FINAL CHECKLIST BEFORE SUBMISSION

- [ ] All 6 models trained successfully
- [ ] All 6 metrics calculated for each model
- [ ] Streamlit app tested locally
- [ ] Executed on BITS Virtual Lab
- [ ] Screenshot captured
- [ ] GitHub repository created and public
- [ ] All files pushed to GitHub
- [ ] Streamlit app deployed successfully
- [ ] Live app URL tested and works
- [ ] PDF created with all required sections
- [ ] Links in PDF are clickable
- [ ] README content included in PDF
- [ ] File uploaded to Taxila
- [ ] **SUBMITTED** (not draft!)

---

**Good Luck! 🚀**

Remember: You have ONE submission attempt. Make it count!
