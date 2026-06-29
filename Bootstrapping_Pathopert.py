#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix
)

# =========================================================
# CONFIG
# =========================================================
N_BOOTSTRAP = 10000
N_JOBS = -1
RANDOM_SEED = 42

# ---------------------------------------------------------
# STRAIN-LEVEL THRESHOLDS
# ---------------------------------------------------------
THRESHOLDS = {
    "ORIGINAL SEQUENCES": 0.5226,
    "REVERSE COMPLEMENT": 0.5377,
    "SIMULATED SEQUENCES": 0.5477
}


# =========================================================
# LOAD PICKLE
# =========================================================
def load_pkl(path):
    with open(path, "rb") as f:
        return np.array(pickle.load(f))


# =========================================================
# COMPUTE METRICS
# =========================================================
def compute_metrics(y_true, y_scores, threshold):

    y_pred = (y_scores > threshold).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return (
        roc_auc_score(y_true, y_scores),
        average_precision_score(y_true, y_scores),
        matthews_corrcoef(y_true, y_pred),
        f1_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
        specificity,
        accuracy_score(y_true, y_pred),
        fpr,
        fnr
    )


# =========================================================
# SINGLE BOOTSTRAP ITERATION
# =========================================================

def stratified_indices(y, rng):
    y = np.asarray(y)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # fallback: full bootstrap if one class missing
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return rng.choice(len(y), size=len(y), replace=True)

    n_pos = pos_idx.size
    n_neg = neg_idx.size

    boot_pos = rng.choice(pos_idx, size=n_pos, replace=True)
    boot_neg = rng.choice(neg_idx, size=n_neg, replace=True)

    return np.concatenate([boot_pos, boot_neg])

def bootstrap_iteration(
    y_true,
    y_scores,
    threshold,
    seed
):

    rng = np.random.default_rng(seed)

    # stratified bootstrap indices (IMPORTANT CHANGE)
    idx = stratified_indices(y_true, rng)

    y_t = y_true[idx]
    y_s = y_scores[idx]

    # AUROC safety (optional but fine)
    if len(np.unique(y_t)) < 2:
        return None

    return compute_metrics(
        y_t,
        y_s,
        threshold
    )

# =========================================================
# PARALLEL BOOTSTRAP
# =========================================================
def bootstrap_metrics_parallel(
    y_true,
    y_scores,
    threshold,
    n_bootstrap=10000,
    n_jobs=-1,
    seed=42
):

    metric_names = [
        "AUROC",
        "AUPR",
        "MCC",
        "F1",
        "Precision",
        "Recall",
        "Specificity",
        "Accuracy",
        "FPR",
        "FNR"
    ]

    seeds = np.random.SeedSequence(seed).spawn(n_bootstrap)

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10
    )(
        delayed(bootstrap_iteration)(
            y_true,
            y_scores,
            threshold,
            s
        )
        for s in seeds
    )

    # Remove invalid bootstrap samples
    results = np.array([
    r if r is not None else np.full(len(metric_names), np.nan)
    for r in results
]) 
    # ✔ rejection stats
    rejected = np.mean(np.isnan(results).any(axis=1)) * 100
    print(f"Rejected bootstrap samples: {rejected:.2f}%")
    
    
    rejected_per_metric = np.isnan(results).mean(axis=0)
    for name, r in zip(metric_names, rejected_per_metric):
        print(f"{name}: {r*100:.2f}%")


    summary = {}

    for i, metric_name in enumerate(metric_names):

        col = results[:, i]

        vals = col[~np.isnan(col)]


        summary[metric_name] = {
            "mean": np.mean(vals),
            "low": np.percentile(vals, 2.5),
            "high": np.percentile(vals, 97.5)
        }

    return summary


# =========================================================
# EVALUATE CONDITION
# =========================================================
def evaluate_condition(
    name,
    y_true,
    y_scores
):

    threshold = THRESHOLDS[name]

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Threshold used: {threshold:.4f}")

    results = bootstrap_metrics_parallel(
        y_true,
        y_scores,
        threshold=threshold,
        n_bootstrap=N_BOOTSTRAP,
        n_jobs=N_JOBS,
        seed=RANDOM_SEED
    )

    rows = []

    for metric_name, vals in results.items():

        print(
            f"{metric_name:<12}"
            f"{vals['mean']:.4f} "
            f"95% CI "
            f"[{vals['low']:.4f}, {vals['high']:.4f}]"
        )

        rows.append({
            "Condition": name,
            "Threshold": threshold,
            "Metric": metric_name,
            "Mean": round(vals['mean'], 4),
            "CI Lower": round(vals['low'], 4),
            "CI Upper": round(vals['high'], 4)
        })

    return rows


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    all_rows = []

    # -----------------------------------------------------
    # ORIGINAL
    # -----------------------------------------------------
    y_scores_kmer = load_pkl(
        "/home/family/Videos/bootstrap/y_scores_kmer.pkl"
    )

    y_kmer = load_pkl(
        "/home/family/Videos/bootstrap/y_kmer.pkl"
    )

    # -----------------------------------------------------
    # SIMULATED
    # -----------------------------------------------------
    y_scores_sim = load_pkl(
        "/home/family/Videos/bootstrap/y_scores_sim.pkl"
    )

    y_sim = load_pkl(
        "/home/family/Videos/bootstrap/y_sim.pkl"
    )

    # -----------------------------------------------------
    # REVERSE COMPLEMENT
    # -----------------------------------------------------
    rev_y_scores = load_pkl(
        "/home/family/Videos/bootstrap/Rev_y_scores.pkl"
    )

    rev_y = load_pkl(
        "/home/family/Videos/bootstrap/Rev_y.pkl"
    )

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------
    all_rows.extend(
        evaluate_condition(
            "ORIGINAL SEQUENCES",
            y_kmer,
            y_scores_kmer
        )
    )

    all_rows.extend(
        evaluate_condition(
            "SIMULATED SEQUENCES",
            y_sim,
            y_scores_sim
        )
    )

    all_rows.extend(
        evaluate_condition(
            "REVERSE COMPLEMENT",
            rev_y,
            rev_y_scores
        )
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------
    df = pd.DataFrame(all_rows)

    output_csv = (
        "/home/family/Videos/bootstrap/"
        "bootstrap_metrics_results_stratified.csv"
    )

    output_excel = (
        "/home/family/Videos/bootstrap/"
        "bootstrap_metrics_results_stratified.xlsx"
    )

    df.to_csv(output_csv, index=False)
    df.to_excel(output_excel, index=False)

    print("\n" + "=" * 80)
    print("Results saved:")
    print(output_csv)
    print(output_excel)
    print("=" * 80)

