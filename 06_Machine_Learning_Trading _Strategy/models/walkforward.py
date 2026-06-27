# models/walkforward.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

def _proba_or_decision(model, X):
    """
    Return a probability-like score in [0, 1] for class 1.
    """
    # prefer calibrated probabilities if available
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Otherwise, use decision_function and map scores to (0,1)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    # last resort: use predict() as 0/1
    y = model.predict(X)
    return (y == 1).astype(float)

def walk_forward_predict(df: pd.DataFrame, features, model, start_idx: int = 12):
    """
    Expanding-window walk-forward backtest.

    At each month t (starting from start_idx), the model is trained on all
    past months and then used to predict the cross-section at month t.
    Returns the full prediction DataFrame plus overall AUC and Brier score.
    """
    dates = sorted(df["Date"].unique())
    frames = []

    # Loop over months, always training on all past months and testing on the next one
    for i in range(start_idx, len(dates) - 1):
        tr_dates, te_date = dates[:i], dates[i]
        train = df[df["Date"].isin(tr_dates)]
        test  = df[df["Date"] == te_date]
        Xtr, ytr = train[features].values, train["target"].values
        Xte      = test[features].values
        model.fit(Xtr, ytr)
        p = _proba_or_decision(model, Xte)
        frames.append(test.loc[:, ["Date","Ticker","target"]].assign(p_up=p))

    # Concatenate all months together
    pred_df = pd.concat(frames, ignore_index=True)

    # Global metrics over the whole out-of-sample period
    auc   = roc_auc_score(pred_df["target"], pred_df["p_up"])
    brier = brier_score_loss(pred_df["target"], pred_df["p_up"])
    
    return pred_df, auc, brier
