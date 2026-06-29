#!/usr/bin/env python
# coding: utf-8

# In[5]:
import joblib
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from contextlib import contextmanager
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from compare_auc_delong_xu import (
    delong_roc_test,
    delong_roc_variance,
    delong_roc_test_2)

from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score
)


@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(
        joblib.parallel.BatchCompletionCallBack
    ):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = (
        TqdmBatchCompletionCallback
    )

    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

# =========================================================
# SETTINGS
# =========================================================
N_BOOTSTRAP = 5000
RANDOM_SEED = 42
ALPHA = 0.05
THRESHOLDS = {
    "ORIGINAL SEQUENCES": 0.5226,
    "REVERSE COMPLEMENT": 0.5377,
    "SIMULATED SEQUENCES": 0.5477
}



# =========================================================
# LOAD
# =========================================================
def load_pkl(path):
    with open(path, "rb") as f:
        return np.array(pickle.load(f))




# =========================================================
# METRICS
# =========================================================

def compute_metrics(y_true, y_scores, t):

    y_pred = (y_scores > t).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    eps = 1e-12

    tpr = tp / (tp + fn + eps)
    tnr = tn / (tn + fp + eps)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)

    precision = tp / (tp + fp + eps)
    recall = tpr

    return {
        "AUROC": roc_auc_score(y_true, y_scores),
        "AUPR": average_precision_score(y_true, y_scores),

        "MCC": matthews_corrcoef(y_true, y_pred),

        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision,
        "Recall": recall,

        "Specificity": tnr,
        "Accuracy": (tp + tn) / (tp + tn + fp + fn + eps),

        "FPR": fpr,
        "FNR": fnr,
        "TPR": tpr,
    }

def fast_metric(y_true, y_scores, metric, t=None):

    # threshold-free metrics
    if metric == "AUROC":
        return roc_auc_score(y_true, y_scores)

    if metric == "AUPR":
        return average_precision_score(y_true, y_scores)

    y_pred = (y_scores > t).astype(np.int8)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    eps = 1e-12

    if metric == "MCC":
        return matthews_corrcoef(y_true, y_pred)

    if metric == "F1":
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        return 2 * p * r / (p + r + eps)

    if metric == "Precision":
        return tp / (tp + fp + eps)

    if metric == "Recall":
        return tp / (tp + fn + eps)

    if metric == "Specificity":
        return tn / (tn + fp + eps)

    if metric == "Accuracy":
        return (tp + tn) / (tp + tn + fp + fn + eps)

    if metric == "FPR":
        return fp / (fp + tn + eps)

    if metric == "FNR":
        return fn / (fn + tp + eps)

    if metric == "TPR":
        return tp / (tp + fn + eps)

    raise ValueError(f"Unknown metric: {metric}")

# =========================================================
# EFFECT SIZE
# =========================================================
def cohens_h(p1, p2):
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


# =========================================================
# MEAN ABSOLUTE PREDICTION SHIFT
# =========================================================
def mean_absolute_prediction_shift(a, b, paired=True):
    if paired:
        if len(a) != len(b):
            raise ValueError(
                f"MAPS requires paired arrays. Got {len(a)} vs {len(b)}"
            )
        return np.mean(np.abs(a - b))

    # unpaired fallback (distribution-level shift)
    return abs(np.mean(a) - np.mean(b))



# =========================================================
# BOOTSTRAP CI
# =========================================================
def bootstrap_ci(y_true, y_scores, metric, t):
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y_true)

    vals = []
    real = fast_metric(y_true, y_scores, metric, t)

    for _ in tqdm(range(N_BOOTSTRAP), desc=f"Bootstrap {metric}", leave=False):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_scores[idx]

        if len(np.unique(yt)) < 2:
            continue

        vals.append(fast_metric(yt,ys,metric,t))

    if len(vals) == 0:
        return real, np.nan, np.nan

    vals = np.array(vals)

    return real, np.percentile(vals, 2.5), np.percentile(vals, 97.5)



# =========================================================
# FOREST PLOT (FIXED ZERO LINE LOGIC)
# =========================================================
def forest_plot(results, title, is_difference_plot=True):

    labels = [
        k for k in results
        if not (
            np.isnan(results[k][1]) or
            np.isnan(results[k][2])
        )
    ]

    est = [results[k][0] for k in labels]
    low = [results[k][1] for k in labels]
    high = [results[k][2] for k in labels]

    plt.figure(figsize=(10, 6))

    y = np.arange(len(labels))

    for i in range(len(labels)):

        color = "red" if est[i] < 0 else "green"

        plt.plot(
            [low[i], high[i]],
            [y[i], y[i]],
            linewidth=2,
            color=color
        )

        plt.scatter(
            est[i],
            y[i],
            s=60,
            color=color,
            zorder=3
        )

    plt.yticks(y, labels)

    plt.xlabel(
        "Metric Difference (Condition - Original)"
    )

    plt.title(title)

    plt.grid(axis="x", alpha=0.3)

    if is_difference_plot:
        plt.axvline(
            0,
            linestyle="--",
            alpha=0.6,
            color="black"
        )

    plt.tight_layout()

    safe_title = (
        title
        .replace(" ", "_")
        .replace(":", "")
        .replace("/", "_")
    )

    plt.savefig(
        f"{safe_title}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
# =========================================================
# WASSERSTEIN DISTANCE (Distribution Shift)
# =========================================================
def wasserstein_shift(a_scores, b_scores):
    """
    Measures distribution shift between two score distributions.
    Works for UNPAIRED data (e.g., simulated vs original).
    """
    return wasserstein_distance(a_scores, b_scores)

def normalized_wasserstein(a, b):
    std = np.std(np.concatenate([a, b]))
    return wasserstein_distance(a, b) / (std + 1e-12)

def wasserstein_signed(a, b):
    return np.mean(b) - np.mean(a)

from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score

import numpy as np



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


def bootstrap_worker_two_sample(
    y1, s1, y2, s2,
    metric,
    t1, t2,
    seed
):
    rng = np.random.default_rng(seed)

    i1 = stratified_indices(y1, rng)
    i2 = stratified_indices(y2, rng)

    # avoid repeated indexing
    yt1 = np.take(y1, i1)
    ys1 = np.take(s1, i1)

    yt2 = np.take(y2, i2)
    ys2 = np.take(s2, i2)

    if len(np.unique(yt1)) < 2 or len(np.unique(yt2)) < 2:
        return None

    m1 = fast_metric(yt1, ys1, metric, t1)
    m2 = fast_metric(yt2, ys2, metric, t2)

    return m2 - m1


def two_sample_bootstrap_fast(
    y1, s1,
    y2, s2,
    metric,
    t1, t2,
    n_jobs=8
):
    print(f"\nStarting {metric}")
    seeds = np.random.SeedSequence(RANDOM_SEED).spawn(N_BOOTSTRAP)

    real = (
        fast_metric(y2, s2, metric, t2)
        - fast_metric(y1, s1, metric, t1)
    )

    with tqdm_joblib(
    tqdm(
        total=N_BOOTSTRAP,
        desc=f"Unpaired {metric}",
        leave=False
    )
):
        results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        batch_size=32
    )(
        delayed(bootstrap_worker_two_sample)(
            y1, s1, y2, s2,
            metric, t1, t2,
            int(seed.generate_state(1)[0])
        )
        for seed in seeds
    )
    diffs = np.fromiter(
        (x for x in results if x is not None),
        dtype=np.float64
    )

    if diffs.size == 0:
        return real, np.nan, np.nan

    return (
        real,
        np.percentile(diffs, 2.5),
        np.percentile(diffs, 97.5)
    )
# =========================================================
# PAIRED WORKER
# =========================================================

def bootstrap_worker_paired(args):
    y, a, b, metric, t1, t2, seed = args
    rng = np.random.default_rng(seed)

    idx = stratified_indices(y, rng)

    aa = a[idx]
    bb = b[idx]

    m1 = fast_metric(y[idx], aa, metric, t1)
    m2 = fast_metric(y[idx], bb, metric, t2)

    return m2 - m1

# =========================================================
# FAST PAIRED BOOTSTRAP
# =========================================================
def paired_bootstrap(
    y,
    a,
    b,
    metric,
    t1, t2,
    n_jobs=8,
):
    """
    Fast paired bootstrap
    """

    print(f"\nFast paired bootstrap: {metric}")

    real = (
        fast_metric(y, b, metric,t2)
        - fast_metric(y, a, metric,t1)
    )

    seeds = np.random.SeedSequence(RANDOM_SEED).spawn(N_BOOTSTRAP)

    worker_args = [
        (
            y,
            a,
            b,
            metric,
            t1, t2,
            int(seed.generate_state(1)[0]),
        )
        for seed in seeds
    ]

    diffs = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:

        for result in tqdm(
            executor.map(bootstrap_worker_paired, worker_args),
            total=N_BOOTSTRAP,
            desc=f"Paired {metric}",
        ):

            if result is not None:
                diffs.append(result)

    if len(diffs) == 0:
        return real, np.nan, np.nan

    diffs = np.asarray(diffs, dtype=np.float64)

    low = np.percentile(diffs, 2.5)
    high = np.percentile(diffs, 97.5)

    return real, low, high


# =========================================================
# PUBLIC DELONG TEST
# =========================================================
import numpy as np
def delong(y_true, p1, p2):

    log10_p = delong_roc_test(
        y_true,
        p1,
        p2
    )
    z_score, p_value_2 = delong_roc_test_2(
        y_true,
        p1,
        p2
    )

    p_value = float(10 ** log10_p)

    auc1, var1 = delong_roc_variance(
        y_true,
        p1
    )

    auc2, var2 = delong_roc_variance(
        y_true,
        p2
    )

    delta_auc = auc2 - auc1

    

    return {
        "aucs": [float(auc1), float(auc2)],
        "delta_auc": float(delta_auc),
        "variance_model1": float(var1),
        "variance_model2": float(var2),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "p_value_2": float(p_value_2)
    }

def auc_ci(auc, var, alpha=0.95):

    sd = np.sqrt(var)

    lower = stats.norm.ppf(
        (1 - alpha) / 2,
        loc=auc,
        scale=sd
    )

    upper = stats.norm.ppf(
        1 - (1 - alpha) / 2,
        loc=auc,
        scale=sd
    )

    return (
        max(0.0, lower),
        min(1.0, upper)
    )
# =====================================================
# PRACTICAL SIGNIFICANCE HELPERS
# =====================================================
def practical_effect_label(delta):

    ad = abs(delta)

    if ad < 0.001:
        return "negligible"

    elif ad < 0.005:
        return "very small"

    elif ad < 0.01:
        return "small"

    elif ad < 0.02:
        return "moderate"

    else:
        return "large"


# In[ ]:


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS PIPELINE")
    print("="*80)

    cache = {}

    print("\nLoading data...")

    y_o = load_pkl("/home/family/Videos/bootstrap/y_kmer.pkl")
    s_o = load_pkl("/home/family/Videos/bootstrap/y_scores_kmer.pkl")

    y_s = load_pkl("/home/family/Videos/bootstrap/y_sim.pkl")
    s_s = load_pkl("/home/family/Videos/bootstrap/y_scores_sim.pkl")

    y_r = load_pkl("/home/family/Videos/bootstrap/Rev_y.pkl")
    s_r = load_pkl("/home/family/Videos/bootstrap/Rev_y_scores.pkl")
    print(type(s_o))
    print(type(s_s))

    print(s_o.shape)
    print(s_s.shape)

    print(s_o.dtype)
    print(s_s.dtype)

    # =====================================================
    # DISTRIBUTION SHIFT ANALYSIS
    # =====================================================
    maps_rev = mean_absolute_prediction_shift(
        s_o,
        s_r,
        paired=True
    )

    maps_sim = mean_absolute_prediction_shift(
        s_o,
        s_s,
        paired=False
    )

    # -----------------------------------------------------
    # Simulated Reads
    # -----------------------------------------------------
    w_sim = wasserstein_distance(s_o, s_s)

    w_sim_norm = normalized_wasserstein(s_o, s_s)

    w_sim_dir = wasserstein_signed(s_o, s_s)

    # -----------------------------------------------------
    # Reverse Complement
    # -----------------------------------------------------
    w_rev = wasserstein_distance(s_o, s_r)

    w_rev_norm = normalized_wasserstein(s_o, s_r)

    w_rev_dir = wasserstein_signed(s_o, s_r)

    print("\n" + "-"*80)
    print("MEAN ABSOLUTE PREDICTION SHIFT")
    print("-"*80)

    print(f"Reverse Complement MAPS : {maps_rev:.6f}")
    print(f"Simulated Reads MAPS    : {maps_sim:.6f}")

    if maps_rev < 0.01:
        print("✓ Very stable predictions under reverse complement")

    elif maps_rev < 0.05:
        print("⚠ Moderate prediction shift under reverse complement")

    else:
        print("✗ High prediction instability under strand flip")

    print(f"Simulated Wasserstein Distance      : {w_sim:.6f}")
    print(f"Simulated Normalized Wasserstein    : {w_sim_norm:.6f}")
    print(f"Simulated Directional Shift (mean)  : {w_sim_dir:+.6f}")

    if w_sim_norm < 0.2:
        print("✓ Minimal distribution shift (robust model)")

    elif w_sim_norm < 0.5:
        print("⚠ Moderate distribution shift")

    else:
        print("✗ Strong distribution shift (domain sensitivity)")

    print("\n" + "-"*80)
    print("REVERSE COMPLEMENT SHIFT ANALYSIS")
    print("-"*80)

    print(f"RevComp Wasserstein Distance     : {w_rev:.6f}")
    print(f"RevComp Normalized Wasserstein   : {w_rev_norm:.6f}")
    print(f"RevComp Directional Shift        : {w_rev_dir:+.6f}")

    if w_rev_norm < 0.1:
        print("✓ Highly invariant under reverse complement")

    elif w_rev_norm < 0.3:
        print("⚠ Mild strand sensitivity")

    else:
        print("✗ Strong strand bias detected")

    # =====================================================
    # SETTINGS
    # =====================================================
    metrics = [
    "AUROC",
    "AUPR",
    "MCC",
    "F1",
    "Precision",
    "Recall",
    "Specificity",
    "Accuracy",
    "FPR",
    "FNR",
    "TPR"]
    
    effect_metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "FPR",
    "FNR",
    "TPR",
    "F1"
]

    print(
        f"Sample sizes: "
        f"Original={len(y_o):,}, "
        f"Simulated={len(y_s):,}, "
        f"RevComp={len(y_r):,}"
    )

    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"Significance level: α={ALPHA}")

    # =====================================================
    # SIMULATED READS BOOTSTRAP
    # =====================================================
    print("\n" + "-"*80)
    print("SIMULATED READS (Distribution Shift - Unpaired)")
    print("-"*80)

    sim_results = {}

    for m in tqdm(metrics, desc="Simulated"):

        sim_results[m] = two_sample_bootstrap_fast(
            y_o,
            s_o,
            y_s,
            s_s,
            m,
            t1=THRESHOLDS["ORIGINAL SEQUENCES"],
            t2=THRESHOLDS["SIMULATED SEQUENCES"],
            n_jobs=8
        )

    # =====================================================
    # REVERSE COMPLEMENT BOOTSTRAP
    # =====================================================
    print("\n" + "-"*80)
    print("REVERSE COMPLEMENT (Invariance - Paired)")
    print("-"*80)

    rev_results = {}

    for m in tqdm(metrics, desc="Reverse"):

        rev_results[m] = paired_bootstrap(
            y_o,
            s_o,
            s_r,
            m,
            t1=THRESHOLDS["ORIGINAL SEQUENCES"],
            t2=THRESHOLDS["REVERSE COMPLEMENT"],
            n_jobs=8
        )

    # =====================================================
    # DELONG TEST
    # =====================================================
    res = delong(y_o, s_o, s_r)

    print("\nDELONG TEST")
    print("-" * 60)

    print(f"Original AUROC : {res['aucs'][0]:.6f}")
    print(f"RevComp AUROC  : {res['aucs'][1]:.6f}")
    print(f"Δ AUROC        : {res['delta_auc']:+.6f}")
    print(
    f"Original Variance : "
    f"{res['variance_model1']:.6e}"
)

    print(
    f"RevComp Variance  : "
    f"{res['variance_model2']:.6e}"
)
    print(f"Z-score        : {res['z_score']:.6f}")
    print(f"P-value        : {res['p_value']:.6e}")
    print(f"P-value_2        : {res['p_value_2']:.6e}")

    if res["p_value"] < ALPHA:
        print("✓ Significant AUROC difference")
        

    else:
        print("✗ No significant AUROC difference")
    
    if res["p_value_2"] < ALPHA:
        print("✓ Significant AUROC difference")
        

    else:
        print("✗ No significant AUROC difference")
    
    ci1 = auc_ci(
    res["aucs"][0],
    res["variance_model1"]
)

    ci2 = auc_ci(
    res["aucs"][1],
    res["variance_model2"]
)   
    print(
    f"Original 95% CI : "
    f"[{ci1[0]:.6f}, {ci1[1]:.6f}]"
)

    print(
    f"RevComp 95% CI  : "
    f"[{ci2[0]:.6f}, {ci2[1]:.6f}]"
)
    # =====================================================
    # INTERPRETATION
    # =====================================================
    print("\n" + "-"*80)
    print(f"INTERPRETATION (α={ALPHA})")
    print("-"*80)

    # -----------------------------------------------------
    # Simulated interpretation table
    # -----------------------------------------------------
    print("\nSIMULATED READS (Simulated vs Original):")

    sim_table = []

    for m in metrics:

        diff, low, high = sim_results[m]

        if np.isnan(low) or np.isnan(high):

            sig = "⚠ INSUFFICIENT DATA"

        elif low > 0:

            sig = "✓ Simulated HIGHER (significant)"

        elif high < 0:

            sig = "✓ Simulated LOWER (significant)"

        else:

            sig = "✗ No significant difference"

        print(f"  {m:12}: {sig}  (Δ={diff:+.4f})")

        sim_table.append({
            "Metric": m,
            "Delta": diff,
            "CI_Low": low,
            "CI_High": high,
            "Interpretation": sig,
            "Practical_Effect":
                practical_effect_label(abs(diff))
        })

    sim_df_2 = pd.DataFrame(sim_table)

    # -----------------------------------------------------
    # Reverse complement interpretation table
    # -----------------------------------------------------
    print("\nREVERSE COMPLEMENT (RevComp vs Original):")

    rev_table = []

    for m in metrics:

        diff, low, high = rev_results[m]

        if np.isnan(low) or np.isnan(high):

            sig = "⚠ INSUFFICIENT DATA"

        elif low > 0:

            sig = "✓ RevComp HIGHER (significant)"

        elif high < 0:

            sig = "✓ RevComp LOWER (significant)"

        else:

            sig = "✗ No significant difference"

        print(f"  {m:12}: {sig}  (Δ={diff:+.4f})")

        rev_table.append({
            "Metric": m,
            "Delta": diff,
            "CI_Low": low,
            "CI_High": high,
            "Interpretation": sig,
            "Practical_Effect":
                practical_effect_label(abs(diff))
        })

    rev_df_2 = pd.DataFrame(rev_table)

    # =====================================================
    # EFFECT SIZE (COHEN'S h)
    # =====================================================
    print("\n" + "-"*80)
    print("EFFECT SIZE (Cohen's h - Simulated vs Original)")
    print("-"*80)

    metrics_o = compute_metrics(y_o, s_o, THRESHOLDS["ORIGINAL SEQUENCES"])
    metrics_s = compute_metrics(y_s, s_s, THRESHOLDS["SIMULATED SEQUENCES"])


    effect_sim_rows = []

    for m in effect_metrics:

        p1 = metrics_o[m]
        p2 = metrics_s[m]

        h = abs(cohens_h(p1, p2))

        if h < 0.2:
            effect = "negligible"

        elif h < 0.5:
            effect = "small"

        elif h < 0.8:
            effect = "medium"

        else:
            effect = "large"

        print(f"{m:12}: Cohen's h = {h:.4f} ({effect})")

        effect_sim_rows.append({
            "Metric": m,
            "Cohens_h": h,
            "Effect_Size": effect
        })

    effect_sim_df_2 = pd.DataFrame(effect_sim_rows)

    # -----------------------------------------------------
    # RevComp effect sizes
    # -----------------------------------------------------
    print("\nEFFECT SIZE (Cohen's h - RevComp vs Original)")

    metrics_r = compute_metrics(y_r, s_r, THRESHOLDS["REVERSE COMPLEMENT"])

    effect_rev_rows = []

    for m in metrics:

        p1 = metrics_o[m]
        p2 = metrics_r[m]

        h = abs(cohens_h(p1, p2))

        effect = practical_effect_label(h)

        print(f"{m:12}: Cohen's h = {h:.4f}")

        effect_rev_rows.append({
            "Metric": m,
            "Cohens_h": h,
            "Effect_Size": effect
        })

    effect_rev_df_2 = pd.DataFrame(effect_rev_rows)

    # =====================================================
    # SUMMARY
    # =====================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    sig_sim = sum(
        1 for m in metrics
        if not np.isnan(sim_results[m][1])
        and (
            sim_results[m][1] > 0
            or sim_results[m][2] < 0
        )
    )

    sig_rev = sum(
        1 for m in metrics
        if not np.isnan(rev_results[m][1])
        and (
            rev_results[m][1] > 0
            or rev_results[m][2] < 0
        )
    )

    print(f"\nSignificant differences (95% CI excludes 0):")

    print(
        f"  Simulated Reads: "
        f"{sig_sim}/{len(metrics)} metrics"
    )

    print(
        f"  Reverse Complement: "
        f"{sig_rev}/{len(metrics)} metrics"
    )

    # =====================================================
    # CONCLUSIONS
    # =====================================================
    print("\n" + "-"*80)
    print("CONCLUSIONS")
    print("-"*80)

    if sig_rev == 0:

        rev_conclusion = (
            "✓ REVERSE COMPLEMENT: "
            "Model is INVARIANT to strand orientation"
        )

    else:

        rev_conclusion = (
            f"⚠ REVERSE COMPLEMENT: "
            f"Model shows STRAND BIAS "
            f"in {sig_rev} metrics"
        )

    print(rev_conclusion)

    if sig_sim <= 2:

        sim_conclusion = (
            "✓ SIMULATED READS: "
            "Model is ROBUST to simulated sequencing errors"
        )

    else:

        sim_conclusion = (
            f"⚠ SIMULATED READS: "
            f"Model shows SENSITIVITY "
            f"to sequencing errors in {sig_sim} metrics"
        )

    print(sim_conclusion)

    # =====================================================
    # PRACTICAL INTERPRETATION
    # =====================================================
    print("\n" + "-"*80)
    print("PRACTICAL INTERPRETATION")
    print("-"*80)

    rev_auc_delta = abs(res["delta_auc"])

    if (
        sig_rev == 0
        or (
            rev_auc_delta < 0.001
            and maps_rev < 0.01
        )
    ):

        rev_practical = (
            "✓ REVERSE COMPLEMENT: "
            "Model is highly invariant "
            "to strand orientation"
        )

    else:

        rev_practical = (
            f"⚠ REVERSE COMPLEMENT: "
            f"Minor strand-dependent effects detected "
            f"({practical_effect_label(rev_auc_delta)} magnitude)"
        )

    print(rev_practical)

    #sim_auc_delta = abs(sim_results["AUROC"][0])
    sim_auc_delta = abs(
    compute_metrics(y_o, s_o, THRESHOLDS["ORIGINAL SEQUENCES"])["AUROC"]
    - compute_metrics(y_s, s_s, THRESHOLDS["SIMULATED SEQUENCES"])["AUROC"]
)

    if (
        sig_sim <= 2
        and sim_auc_delta < 0.01
        and w_sim_norm < 0.05
    ):

        sim_practical = (
            "✓ SIMULATED READS: "
            "Model is robust to sequencing noise "
            "and distribution shift"
        )

    else:

        sim_practical = (
            f"⚠ SIMULATED READS: "
            f"Performance degradation observed "
            f"({practical_effect_label(sim_auc_delta)} magnitude)"
        )

    print(sim_practical)

    # =====================================================
    # SAVE TABLES
    # =====================================================
    print("\n" + "-"*80)
    print("SAVING RESULT TABLES")
    print("-"*80)

    # -----------------------------------------------------
    # Bootstrap results
    # -----------------------------------------------------
    sim_df_2.to_csv(
        "simulated_bootstrap_results_2.csv",
        index=False
    )

    rev_df_2.to_csv(
        "reverse_complement_bootstrap_results_2.csv",
        index=False
    )

    print("✓ Saved bootstrap result tables")

    # -----------------------------------------------------
    # Effect sizes
    # -----------------------------------------------------
    effect_sim_df_2.to_csv(
        "effect_sizes_simulated_2.csv",
        index=False
    )

    effect_rev_df_2.to_csv(
        "effect_sizes_revcomp_2.csv",
        index=False
    )

    print("✓ Saved effect size tables")

    # -----------------------------------------------------
    # Wasserstein results
    # -----------------------------------------------------
    wasserstein_df_2 = pd.DataFrame([
        {
            "Comparison": "Simulated_vs_Original",
            "MAPS": maps_sim,
            "Wasserstein": w_sim,
            "Normalized_Wasserstein": w_sim_norm,
            "Directional_Shift": w_sim_dir,
        },
        {
            "Comparison": "RevComp_vs_Original",
            "MAPS": maps_rev,
            "Wasserstein": w_rev,
            "Normalized_Wasserstein": w_rev_norm,
            "Directional_Shift": w_rev_dir,
        }
    ])

    wasserstein_df_2.to_csv(
        "wasserstein_shift_results_2.csv",
        index=False
    )

    print("✓ Saved wasserstein results")

    # -----------------------------------------------------
    # Delong results
    # -----------------------------------------------------
    delong_df_2 = pd.DataFrame([{
        "Original_AUROC": res["aucs"][0],
        "RevComp_AUROC": res["aucs"][1],
        "Delta_AUROC": res["delta_auc"],
        "variance_model1": res["variance_model1"],
        "variance_model2": res["variance_model2"],
        "Z_score": res["z_score"],
        "P_value": res["p_value"],
        "P_value_2": res["p_value_2"],
        "Significant":
            res["p_value"] < ALPHA,
        "Practical_Effect":
            practical_effect_label(
                abs(res["delta_auc"])
            )
    }])

    delong_df_2.to_csv(
        "delong_results_2.csv",
        index=False
    )

    print("✓ Saved delong results")

    # -----------------------------------------------------
    # Summary table
    # -----------------------------------------------------
    summary_df_2 = pd.DataFrame([{
        "Significant_Simulated_Metrics": sig_sim,
        "Significant_RevComp_Metrics": sig_rev,
        "Total_Metrics": len(metrics),
        "Reverse_Conclusion": rev_conclusion,
        "Simulated_Conclusion": sim_conclusion,
        "Reverse_Practical": rev_practical,
        "Simulated_Practical": sim_practical,
    }])

    summary_df_2.to_csv(
        "summary_results_2.csv",
        index=False
    )

    print("✓ Saved summary results")

    # =====================================================
    # FOREST PLOTS
    # =====================================================
    print("\n" + "-"*80)
    print("GENERATING FOREST PLOTS")
    print("-"*80)

    forest_plot(
        sim_results,
        "Simulated Reads Difference_2",
        is_difference_plot=True
    )

    forest_plot(
        rev_results,
        "Reverse Complement Difference_2",
        is_difference_plot=True
    )

    # =====================================================
    # FINISHED
    # =====================================================
    print("\n" + "="*80)
    print("✔ STATISTICAL ANALYSIS COMPLETE")
    print("="*80)


# In[ ]:





# In[ ]:




