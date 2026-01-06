Hybrid Predictive Maintenance Using NASA CMAPSS Dataset
 Project Overview
Predictive maintenance aims to anticipate equipment failures before they occur, reducing downtime and maintenance cost.
This project implements a hybrid machine learning pipeline using NASA’s CMAPSS turbofan engine dataset to detect degradation stages, predict remaining useful time, and generate a maintenance risk score.
Unlike traditional approaches that rely on fixed RUL thresholds, this system automatically learns degradation stages using clustering and combines classification + regression for robust decision-making.
________________________________________
 Objectives
•	Identify multi-stage degradation behavior in engines
•	Predict current health stage of an engine
•	Estimate time-to-next-stage / time-to-failure
•	Compute a maintenance risk score for proactive alerts
•	Evaluate generalization across operating conditions
________________________________________
 Dataset
•	Source: NASA CMAPSS Turbofan Engine Dataset
•	Subsets Used: FD001, FD002, FD003, FD004
•	Features:
o	Operational settings (op1, op2, op3)
o	21 sensor measurements
•	Target Concepts:
o	Degradation stages (unsupervised)
o	Remaining time / risk score
________________________________________
 System Architecture (Hybrid Pipeline)
Phase 1: Degradation Stage Labeling (Unsupervised)
•	Algorithm: K-Means Clustering
•	Input: Selected normalized sensor features
•	Output: Automatically labeled stages such as:
o	Healthy
o	Early Degradation
o	Severe Degradation
•	Benefit: Removes reliance on arbitrary RUL thresholds
________________________________________
Phase 2: Degradation Stage Classification (Supervised)
•	Models used: Random Forest / XGBoost / MLP (configurable)
•	Task: Predict degradation stage at each time step
•	Evaluation:
o	Accuracy
o	Precision, Recall, F1-score
o	Confusion Matrix
•	Generalization tested using:
o	FD001 + FD003 → Train
o	FD004 → Test
________________________________________
Phase 3: Time-to-Failure / Time-to-Next-Stage Regression
•	Models: Linear Regression / Gradient Boosting / ML-based Regressors
•	Output:
o	Estimated remaining time
o	Stage transition horizon
•	Trained with engine-wise temporal consistency
________________________________________
Phase 4: Maintenance Risk Scoring
A unified risk score is computed using:
Risk Score = α · Normalized(Failure Probability) 
           + β · Normalized(1 / Predicted Time)
•	High score → Immediate maintenance alert
•	Medium score → Schedule inspection
•	Low score → Normal operation
________________________________________
 Experimental Strategy
•	Per-engine normalization (avoids leakage)
•	Operating condition awareness
•	Cross-dataset validation
•	Multiple dataset combinations:
o	FD001 + FD003
o	FD002 + FD004
o	All datasets combined
________________________________________
 Evaluation Metrics
•	Classification:
o	Accuracy
o	Precision
o	Recall
o	F1-score
•	Regression:
o	MAE
o	RMSE
•	Risk scoring:
o	Threshold-based alert evaluation

PROJECT STRUCTURE:
Hybrid-Predictive-Maintenance/
│
├── data/
│   ├── train_FD002.txt
│   └── train_FD004.txt
│
├── models/
│   ├── stage_classifier.pkl
│   ├── stage_scaler.pkl
│   ├── regressor.pkl
│   └── regressor_scaler.pkl
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── phase1_clustering.py
│   ├── phase2_classification.py
│   ├── phase3_regression.py
│   ├── phase4_risk_scoring.py
│   └── main.py   ✅
│
├── requirements.txt
└── README.md

How to Run
Install Dependencies
pip install -r requirements.txt
To run main code:
python .\src\main.py


