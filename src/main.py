from data_loader import load_cmaps_data
from preprocessing import add_operating_clusters, normalize_sensors_by_cluster
from phase1_clustering import perform_degradation_clustering, SELECTED_SENSORS
from phase2_classification import train_stage_classifier
from phase3_regression import compute_time_to_next_stage, train_regressor
from phase4_risk_scoring import compute_risk_score

import pandas as pd
import joblib
import os

def main():
    print(" Starting Hybrid Predictive Maintenance Pipeline")

    # -----------------------------
    # Load Data
    # -----------------------------
    df2 = load_cmaps_data("data/train_FD002.txt")
    df4 = load_cmaps_data("data/train_FD004.txt")

    df = pd.concat([df2, df4], ignore_index=True)
    print(" Data Loaded:", df.shape)

    # -----------------------------
    # Preprocessing
    # -----------------------------
    df = add_operating_clusters(df)
    df = normalize_sensors_by_cluster(df, SELECTED_SENSORS)
    print(" Preprocessing Completed")

    # -----------------------------
    # Phase 1: Clustering
    # -----------------------------
    df = perform_degradation_clustering(df)
    print(" Degradation Stages Assigned")

    # -----------------------------
    # Phase 2: Classification
    # -----------------------------
    clf, clf_scaler = train_stage_classifier(df, SELECTED_SENSORS)
    print(" Stage Classifier Trained & Saved")

    # -----------------------------
    # Phase 3: Regression
    # -----------------------------
    df_reg = compute_time_to_next_stage(df)
    regressor, reg_scaler = train_regressor(df_reg, SELECTED_SENSORS)
    print(" Regressor Trained & Saved")

    # -----------------------------
    # Phase 4: Risk Scoring (Sample)
    # -----------------------------
    X_sample = clf_scaler.transform(df[SELECTED_SENSORS].iloc[:100])
    risk_scores = compute_risk_score(clf, regressor, X_sample)

    print(" Sample Risk Scores:", risk_scores[:10])
    print(" Pipeline Completed Successfully")

if __name__ == "__main__":
    main()
