import numpy as np

def compute_risk_score(clf, regressor, X, stage_label="Critical"):
    failure_idx = list(clf.classes_).index(stage_label)
    failure_prob = clf.predict_proba(X)[:, failure_idx]
    time_left = regressor.predict(X)

    raw_score = failure_prob / (time_left + 1e-6)
    norm_score = (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min() + 1e-6)

    return norm_score
