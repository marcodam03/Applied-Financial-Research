"""
Capacity: how much money can the strategy run before trading costs eat the alpha?
(docs/STRATEGY.md)

Run from the project folder:
    python scripts/run_capacity.py                 (final strategy: $100m to $2.5bn per side)
    python scripts/run_capacity.py --scenarios     (cost model: spread only, sqrt impact,
                                                    $10m to $2.5bn, 5 days to trade)
    python scripts/run_capacity.py --signal qvi_intan_full --config configs/annual.yaml
                                   (cost-model sensitivity: spread only, sqrt impact, $10m-$2.5bn)

Same backtest as run_signal.py (same universe, sort, spread and price-impact model);
only the fund size changes: $100m (the main runs), $500m, $1bn and $2.5bn per leg in
2025-12 dollars, moved back in time with the market's dollar volume. Bigger fund =
bigger share of each stock's daily volume = more price impact.
Value-weighted only. Nothing in results/signals/ is overwritten: the table is saved
in results/capacity/.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

import run_signal as rs  # noqa: E402
from src.data.panel import load_spreads  # noqa: E402
from src.evaluation.alpha import load_factors, regress  # noqa: E402
from src.signals import build_panel, load_signal  # noqa: E402

SIZES = [100, 500, 1000, 2500]   # $m per leg in 2025-12
# --scenarios: cost-model sensitivity at the same time (each = one full backtest)
SCENARIOS = {
    "spread only":         {"price_impact": False},
    "linear $10m":         {"price_impact": True, "aum_musd": 10},
    "linear $100m (main)": {"price_impact": True, "aum_musd": 100},
    "sqrt $100m":          {"price_impact": True, "aum_musd": 100, "impact_shape": "sqrt"},
    "linear $500m":        {"price_impact": True, "aum_musd": 500},
    "linear $1bn":         {"price_impact": True, "aum_musd": 1000},
    "linear $2.5bn":       {"price_impact": True, "aum_musd": 2500},
    "linear $100m, 5 days to trade": {"price_impact": True, "aum_musd": 100, "execution_days": 5},
}
MODELS = ["CAPM", "FF5", "FF6", "q5"]

pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="qvi_timely")
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--scenarios", action="store_true",
                    help="cost-model sensitivity (spread only, sqrt impact, $10m..$2.5bn, 5 days) "
                         "instead of the fund sizes only")
    args = ap.parse_args()
    cfg = yaml.safe_load((BASE_DIR / args.config).read_text())
    train, valid = tuple(cfg["periods"]["train"]), rs.cut_at_holdout(tuple(cfg["periods"]["validation"]))
    periods = {"train": train, "validation": valid, "train+validation": (train[0], valid[1])}
    spec = load_signal(args.signal)
    spread = f"{spec.LONG}-{spec.SHORT}"
    factors = load_factors(BASE_DIR / cfg["data"]["factors"])
    portfolio_cfg = {**cfg["portfolio"], "n_portfolios": getattr(spec, "N_PORTFOLIOS", cfg["portfolio"]["n_portfolios"])}
    opts = rs.sort_options(cfg["universe"], portfolio_cfg)

    print(f"Loading the panel and computing '{args.signal}'...")
    panel = build_panel(spec)
    spreads = load_spreads()
    rows = []
    if args.scenarios:
        runs = SCENARIOS
    else:
        runs = {size: {"aum_musd": size} for size in SIZES}
    key = "cost scenario" if args.scenarios else "fund $m per leg"
    for size, change in runs.items():
        costs_cfg = {**(cfg.get("costs") or {}), **change}
        impact, label = rs.cost_settings(panel, spreads, costs_cfg)
        print(f"  {label}")
        base = rs.backtest(panel, args.signal, spec, opts, ["vw"], spreads, train[0], valid[1], impact)
        r = base["vw"]
        cost = base["costs_vw"][[spec.LONG, spec.SHORT]].sum(axis=1)
        for period, (s, e) in periods.items():
            net = r[f"{spread} net"].loc[s:e]
            to = (base["turnover_vw"][spec.LONG] + base["turnover_vw"][spec.SHORT]).loc[s:e] / 2
            row = {key: size, "period": period, "Turnover %": to.mean() * 100,
                   "Gross %": r[spread].loc[s:e].mean() * 100,
                   "Net %": net.mean() * 100, "Net t": net.mean() / net.std() * len(net) ** 0.5,
                   "Cost %": cost.loc[s:e].mean() * 100}
            for m in MODELS:
                res = regress(net, factors, m, long_short=True, nw_lags=6)
                row[f"{m} net alpha %"], row[f"{m} t"] = res["alpha %"], res["t(alpha)"]
            rows.append(row)
    table = pd.DataFrame(rows).set_index(["period", key])
    table = table.loc[[p for p in periods if p in table.index.get_level_values(0)]]
    out = BASE_DIR / "results" / "capacity"
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_scenarios" if args.scenarios else ""
    table.to_csv(out / f"{args.signal}_{Path(args.config).stem}{suffix}.csv")
    print(f"\n===== Capacity of {args.signal}, value-weighted, config {args.config} =====")
    print("Net alphas are the regression of the NET spread (after costs) on each model.")
    print(table.to_string())
    print(f"\nSaved in {out}")


if __name__ == "__main__":
    main()
