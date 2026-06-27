# report/plots.py
import os
import matplotlib.pyplot as plt
import numpy as np

def _ensure_dir(path):
    """
    Create directory if it does not exist.
    """

    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def plot_equity(bt, title, savepath=None, show=True):
    """
    Plot cumulative equity curves for the available strategies.
    """
    # plot ML / EW / Momentum curves
    plt.figure(figsize=(8,4))
    if "cum_ret_ml" in bt:
        plt.plot(bt.index, bt["cum_ret_ml"], label="ML Top 10%")
    if "cum_ret_ew" in bt:
        plt.plot(bt.index, bt["cum_ret_ew"], label="Equal-Weight")
    if "cum_ret_mom" in bt:
        plt.plot(bt.index, bt["cum_ret_mom"], label="Momentum 6M Top 10%")

    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # save to file or show
    if savepath:
        _ensure_dir(savepath)
        plt.savefig(savepath, dpi=150)
    if show and not savepath:
        plt.show()
    plt.close()

def plot_underwater(cum_series, savepath=None, show=True):
    """
    Plot underwater (drawdown) curve from a cumulative return series.
    """
    # compute drawdown = cumulative / running peak - 1
    peak = cum_series.cummax()
    dd = (cum_series / peak) - 1.0

    plt.figure(figsize=(8,3))
    plt.plot(dd.index, dd.values)
    plt.title("Underwater (Drawdown)"); plt.grid(True); plt.tight_layout()

    # save to file or show
    if savepath:
        _ensure_dir(savepath)
        plt.savefig(savepath, dpi=150)
    if show and not savepath:
        plt.show()
    plt.close()

def plot_calibration(cal_df, savepath=None, show=True):
    """
    Plot calibration curve comparing predicted probabilities vs empirical rates.
    """
    # choose robust column names (bin midpoint or predicted mean)
    x = cal_df["bin_mid"] if "bin_mid" in cal_df.columns else cal_df["pred_mean"]
    y = cal_df["emp_rate"]

    plt.figure(figsize=(4,4))
    plt.plot([0,1],[0,1], linestyle="--", label="Ideal")
    plt.scatter(x, y, s=25, label="Model")
    plt.xlabel("Predicted probability"); plt.ylabel("Empirical frequency")
    plt.title("Calibration"); plt.legend(); plt.grid(True); plt.tight_layout()

    # save figure or show
    if savepath:
        _ensure_dir(savepath)
        plt.savefig(savepath, dpi=150)
    if show and not savepath:
        plt.show()
    plt.close()

def plot_feature_importance(model, feature_names,
                            savepath="report/outputs/feat_importance.png"):
    """
    Plot feature importances for models exposing the feature_importances_ attribute.
    """

    # final estimator inside a Pipeline or standalone model
    final_est = model[-1] if hasattr(model, "__getitem__") else model

    if not hasattr(final_est, "feature_importances_"):
        print("Model has no feature_importances_ attribute – skipping plot.")
        return

    imp = np.array(final_est.feature_importances_)
    idx = np.argsort(imp)[::-1] # sort by importance descending

    plt.figure(figsize=(6, 4))
    plt.bar(range(len(imp)), imp[idx])
    plt.xticks(range(len(imp)),
               [feature_names[i] for i in idx],
               rotation=45, ha="right")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    plt.close()
