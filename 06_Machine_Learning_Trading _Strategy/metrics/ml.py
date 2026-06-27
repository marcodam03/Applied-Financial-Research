# metrics/ml.py
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score
)
import pandas as pd
import numpy as np

def evaluate_probs(y_true, p):
    """
    Compute a standard metrics for probabilistic binary classification.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p)

    return {
        "AUC": roc_auc_score(y_true,p),
        "Brier": brier_score_loss(y_true,p),
        "Acc@0.5": accuracy_score(y_true, p>0.5),
        "Prec@0.5": precision_score(y_true, p>0.5, zero_division=0),
        "Rec@0.5":  recall_score(y_true, p>0.5, zero_division=0),
    }

def calib_points(y_true, p_pred, n_bins=10):
    """
    Creates quantile bins of predicted probabilities and compares them
    with the real frequency of y=1 in each bin.
    """
    df = pd.DataFrame({
        "y": np.asarray(y_true, dtype=float),
        "p": np.asarray(p_pred, dtype=float)
    })

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    grp = df.groupby("bin", observed=True)

    out = grp.agg(
        emp_rate=("y", "mean"),
        pred_mean=("p", "mean"),
        count=("y", "size")
    ).reset_index(drop=True)

    out["bin_mid"] = out["pred_mean"]

    return out[["bin_mid", "emp_rate", "pred_mean", "count"]]

def confusion_at_threshold(y, p, thr=0.5):
    """
    Return confusion matrix counts (TP, FP, TN, FN) at a given probability threshold.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    yhat = (p >= thr).astype(int)

    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())

    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
