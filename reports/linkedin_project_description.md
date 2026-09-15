# LinkedIn Project Description (Featured / Projects Section)

### Title
Disease Diagnosis Prediction — End-to-End Machine Learning API

### Short Summary
Built a production-structured machine learning application for heart disease risk classification, combining exploratory data analysis, leakage-safe model training, FastAPI REST service, and Docker containerization.

### Full Description
Excited to share my latest machine learning project: **Disease Diagnosis Prediction Using Clinical Data and Classification Models**.

This project focuses on building a complete, leakage-free machine learning workflow for clinical binary classification ($N=918$ observations):

🔹 **Data Preprocessing & EDA**: Implemented median imputation for suspicious zero-cholesterol records, categorical one-hot encoding, and feature scaling strictly isolated within training folds to prevent data leakage.
🔹 **Model Evaluation & Tuning**: Evaluated 5 classification algorithms (Logistic Regression, KNN, Decision Tree, Random Forest, SVM). Optimized a Linear Support Vector Machine (`SVC`) via 5-fold Stratified `GridSearchCV`, achieving **84.78% accuracy**, **87.25% recall**, **0.8641 F1**, and **0.9295 ROC-AUC** on a held-out test set ($N=184$).
🔹 **Model Persistence & Prediction Pipeline**: Persisted the complete scikit-learn pipeline to joblib (`final_model.joblib`) and developed a reusable inference engine supporting single and batch predictions.
🔹 **FastAPI & Docker Deployment**: Built RESTful API endpoints (`/predict`, `/predict/batch`) with Pydantic validation and containerized the service using Docker Compose and Uvicorn.
🔹 **Testing & Validation**: Verified API health, schema validation, and numerical reproducibility across 142 automated tests (`pytest`).

📁 Check out the complete technical report and codebase on GitHub!

*Disclaimer: Research prototype for machine learning demonstration; not a clinically validated diagnostic system.*
