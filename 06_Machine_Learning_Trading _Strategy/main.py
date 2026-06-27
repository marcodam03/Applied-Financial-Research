"""
Main runner for Marco D’Amico — Advanced Data Science Project
Machine Learning–Based Stock Selection and Trading Strategy
------------------------------------------------------------
This script runs the full pipeline:
1. Download data
2. Build features
3. Train all ML models (walk-forward)
4. Select best model
5. Backtest portfolio
6. Generate figures, tables and robustness checks
"""

import os
import pandas as pd

from settings import FEATURES_BASE, TOP_FRACS, COSTS, SUBPERIODS, SEED
from data.loader import download_prices
from features.build_dataset import build_ml_dataset
from models.models import MODELS
from models.walkforward import walk_forward_predict

from portfolio.backtest import (
    backtest_long_top,
    compute_turnover,
    sensitivity_grid,
)

from metrics.ml import calib_points
from report.tables import perf_table
from report.plots import (
    plot_equity,
    plot_underwater,
    plot_calibration,
    plot_feature_importance,
)

from tabulate import tabulate

def print_section(title):
    """
    Separator for console output.
    """
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")

def print_df_nice(df, title=None, show_index=True):
    """
    Formatted DataFrame print using tabulate.
    """
    if title:
        print_section(title)

    df_to_print = df.copy()

    # round float columns for readability
    float_cols = df_to_print.select_dtypes(include=["float", "float64", "float32"]).columns
    df_to_print[float_cols] = df_to_print[float_cols].round(4)

    # show strategy names as first column
    if show_index:
        df_to_print = df_to_print.reset_index().rename(columns={"index": "Strategy"})
        showindex = False
    else:
        showindex = False

    print(tabulate(df_to_print, headers="keys", tablefmt="psql", showindex=showindex))
    print()

def main():
    # create output directory if it does not exist
    os.makedirs("report/outputs", exist_ok=True)

    # === STEP 1. Data & Features ===
    # load or download raw price data
    if not os.path.exists("data/sp500_prices.csv"):
        print("Downloading price data...")
        download_prices(start="2020-11-01", end="2025-11-01")
    else:
        print("Price data already available.")

    # load or build ML dataset (features + next-month returns)
    if not os.path.exists("data/ml_dataset.csv"):
        print("Building feature dataset...")
        df_ml = build_ml_dataset()
    else:
        print("ML dataset already exists — loading it.")
        df_ml = pd.read_csv("data/ml_dataset.csv", parse_dates=["Date"])

    # === STEP 2. Train all models (walk-forward) ===
    results = []
    pred_store = {}

    # train each model in walk-forward scheme and store predictions/metrics
    for name, model in MODELS.items():
        print(f"Training {name} ...")
        pred, auc, brier = walk_forward_predict(
            df_ml, FEATURES_BASE, model, start_idx=12
        )
        results.append({"Model": name, "AUC": auc, "Brier": brier})
        pred_store[name] = pred

    # rank models by AUC
    res = pd.DataFrame(results).sort_values("AUC", ascending=False)
    print_df_nice(res, title="MODEL RESULTS (AUC & Brier)")

    best = res.iloc[0]["Model"]
    print(f"\n🏆 Best model selected: {best}")

    # === STEP 3. Build df_pred (attach p_up from best model) ===
    pred_df = pred_store[best]  # columns: Date, Ticker, target, p_up

    df_pred = df_ml.merge(
        pred_df[["Date", "Ticker", "p_up"]],
        on=["Date", "Ticker"],
        how="left",
    )

    # simple sanity check on prediction coverage
    mask = df_pred["p_up"].notna()
    print(
        "Coverage:",
        df_pred.loc[mask, "Date"].min(),
        "→",
        df_pred.loc[mask, "Date"].max(),
    )
    print("Months with predictions:", mask.sum())

    # === STEP 4. Backtest full sample ===
    # use first TOP_FRACS value and second COSTS entry
    top_frac = TOP_FRACS[0] if isinstance(TOP_FRACS, (list, tuple)) else TOP_FRACS
    if isinstance(COSTS, (list, tuple)) and len(COSTS) > 1:
        cost = COSTS[1]
    else:
        cost = 0.0005

    # run long-only ML backtest
    bt = backtest_long_top(df_pred, top_frac=top_frac, cost=cost)

    # compute standard performance metrics (CAGR, Vol, Sharpe, Max DD)
    perf = perf_table(bt)
    print_df_nice(
        perf,
        title="PERFORMANCE TABLE — ML Long-Only vs Benchmarks",
        show_index=True,
    )

    # save performance summary and full backtest
    perf.to_csv("report/outputs/performance_summary.csv")
    bt.to_csv("report/outputs/backtest_full.csv")

    # ML Long-Short strategy (Top vs Bottom) ===
    from portfolio.backtest import backtest_long_short
    from metrics.perf import cagr, ann_vol, sharpe, max_drawdown

    print("\n=== Long-Short ML Strategy (Top vs Bottom) ===")

    # run long–short version of the strategy
    bt_ls = backtest_long_short(df_pred, top_frac=top_frac, cost=cost)

    if bt_ls.empty:
        print_section("LONG-SHORT ML STRATEGY (Top vs Bottom)")
        print("No valid months — skipping.")
    else:
        # save long-short results
        bt_ls.to_csv("report/outputs/backtest_long_short.csv")

        # summarize long-short performance
        ls_metrics = {
            "CAGR":  cagr(1 + bt_ls["ret_long_short"]),
            "Vol":   ann_vol(bt_ls["ret_long_short"]),
            "Sharpe": sharpe(bt_ls["ret_long_short"]),
            "MaxDD": max_drawdown(bt_ls["cum_ret_long_short"]),
        }
        ls_df = pd.DataFrame([ls_metrics])

        print_df_nice(ls_df, title="LONG-SHORT ML STRATEGY (Top 10% vs Bottom 10%)")

        # plot long-short cumulative curve
        from report.plots import plot_equity
        tmp = bt_ls.rename(columns={"cum_ret_long_short": "cum_ret_ml"})

        plot_equity(
            tmp,
            f"Cumulative Return — Long-Short ML (Top {int(top_frac*100)}% vs Bottom {int(top_frac*100)}%)",
            savepath=f"report/outputs/equity_long_short_{best}.png",
        )

    # === STEP 5. Subperiod analysis ===
    print("\n=== Subperiod analysis ===")

    for (s, e) in SUBPERIODS:
        s = pd.to_datetime(s)
        e = pd.to_datetime(e)

        # filter data to selected subperiod
        sub = df_pred[(df_pred["Date"] >= s) & (df_pred["Date"] <= e)]
        if sub["p_up"].notna().sum() == 0:
            print(f"Subperiod {s.date()}–{e.date()}: no predicted months — skipping.")
            continue

        # run long-only backtest inside the subperiod
        bt_sub = backtest_long_top(sub, top_frac=top_frac, cost=cost)
        if bt_sub.empty:
            print(f"Subperiod {s.date()}–{e.date()}: empty backtest — skipping.")
            continue

        # print table of metrics
        title = f"SUBPERIOD {s.date()} – {e.date()}"
        print_df_nice(
            perf_table(bt_sub),
            title=title,
            show_index=True,
        )



    # === STEP 6. Main plots ===
    # equity curve for ML / EW / Momentum
    plot_equity(
        bt,
        f"Cumulative Return — {best}",
        savepath=f"report/outputs/equity_{best}.png",
    )

    # drawdown curve for ML strategy
    plot_underwater(
        bt["cum_ret_ml"],
        savepath=f"report/outputs/underwater_{best}.png",
    )

    # calibration curve (predicted vs empirical probabilities)
    cal = calib_points(
        df_pred.loc[mask, "target"],
        df_pred.loc[mask, "p_up"],
    )
    plot_calibration(cal, savepath=f"report/outputs/calibration_{best}.png")
    cal.to_csv("report/outputs/calibration_points.csv", index=False)

    # === STEP 7. Turnover & sensitivity analysis ===
    # estimate month-to-month turnover of ML top group
    turn = compute_turnover(df_pred, top_frac=top_frac)
    turn.to_csv("report/outputs/turnover.csv")

    # run sensitivity grid for multiple top_frac and cost settings
    sens = sensitivity_grid(df_pred, top_fracs=TOP_FRACS, costs=COSTS, subperiods=SUBPERIODS)
    sens.to_csv("report/outputs/sensitivity.csv", index=False)

    # print a sorted version for readability
    sens_sorted = sens.sort_values(["period", "top_frac", "cost"])
    print_df_nice(
        sens_sorted,
        title="SENSITIVITY GRID — top_frac, cost, period",
        show_index=False,
    )

    # === STEP 8. Feature importance ===
    try:
        # refit best model on full data to extract feature importances
        best_model = MODELS[best]
        X_all = df_ml[FEATURES_BASE].values
        y_all = df_ml["target"].values
        best_model.fit(X_all, y_all)

        plot_feature_importance(
            best_model,
            FEATURES_BASE,
            savepath=f"report/outputs/feat_importance_{best}.png",
        )
    except Exception as e:
        print("Could not plot feature importance:", e)

    # === STEP 9. Save model comparison table ===
    res.to_csv("report/outputs/model_metrics.csv", index=False)


if __name__ == "__main__":
    main()
