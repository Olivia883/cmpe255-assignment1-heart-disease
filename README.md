# cmpe255-assignment1-heart-disease

1. Overview:
This project applies the CRISP‑DM methodology to the Kaggle Heart Disease dataset using an AI coding assistant (ChatGPT Code Interpreter). The goal is to build a predictive model that determines whether a patient is likely to have heart disease based on clinical features.
This assignment includes:
    • Full CRISP‑DM workflow
    • EDA, preprocessing, modeling, evaluation
    • Multiple ML algorithms
    • A YouTube walkthrough
    • Replication of instructor‑provided experiments (Part 2)
   
2. Dataset:
   • Name: Heart Disease Dataset
   • Source: Kaggle
   • Target Variable: target (1 = heart disease, 0 = no heart disease)
   • Type: Binary classification

3. CRISP‑DM Phases

Business Understanding
  - Heart disease is a major health issue. The goal is to build a model that helps identify high‑risk patients.

My paraphrase: Predict heart disease early so doctors can help patients sooner.

Data Understanding
  - Looked at columns, summary stats, missing values, and basic patterns.

Data Preparation
  - Cleaned data, encoded categories, scaled features, and split into train/test sets.

EDA + Visualization
  - Created plots to understand relationships and correlations.

Outlier Analysis
  - Checked for unusual values and decided whether to keep or remove them.

Feature Engineering
  - Selected important features and created any helpful new ones.

Modeling:
Trained multiple models:
  • Logistic Regression
  • Random Forest
  • XGBoost
  • SVM
  • KNN

Evaluation
Compared models using:
  • Accuracy
  • Precision
  • Recall
  • F1
  • ROC‑AUC
  • Confusion Matrix

Final Recommendation
Summarized best model, limitations, and future improvements.

4. Files in This Repository

data/                → dataset
code/                → preprocessing, modeling, evaluation scripts  
images/              → plots and visualizations  
transcripts/         → ChatGPT session  
assignment_part2/    → replicated experiments  
video/               → YouTube link (or text file with link)
