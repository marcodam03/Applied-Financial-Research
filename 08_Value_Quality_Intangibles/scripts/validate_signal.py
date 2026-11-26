"""
Check a signal's portfolios against Ken French's published portfolios.

For published anomalies whose module in src/signals/ sets FRENCH_BENCHMARK
(asset_growth, book_to_market, momentum). The sort follows French's rules: NYSE
decile breakpoints, all NYSE/AMEX/NASDAQ stocks, no price or size filter, and
the full sample. That is on purpose: the question here is "is the code right?",
not "does the signal work?", and French's returns are public anyway.

Run from the project folder:  python scripts/validate_signal.py --signal momentum

If the code is right, each decile should have a correlation of about 0.95 or
more with French's and a similar average return. The spread is a harder test
(the market movement both portfolios share cancels out, small differences
remain), so 0.90 is enough there. Do not expect a perfect match: French uses his
own CRSP/Compustat vintage and, for book equity, extra hand-collected data.
Output: results/validation/<signal>_ew.csv and <signal>_vw.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # lets Python find the "src" folder

from src.data.factors import get_french, read_french_portfolios  # noqa: E402
from src.portfolio.sorts import sort_portfolios  # noqa: E402
from src.signals import build_panel, load_signal  # noqa: E402

START = "1963-07"
MIN_CORR = 0.95         # each portfolio
MIN_CORR_SPREAD = 0.90  # long-short spread
OUT_DIR = BASE_DIR / "results" / "validation"

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")


def compare(ours: pd.DataFrame, french: pd.DataFrame, long_port: int, short_port: int) -> pd.DataFrame:
    """Average return (% per month), correlation and tracking error of each
    portfolio and of the spread, over the months both have."""
    months = ours.dropna(how="all").index.intersection(french.dropna(how="all").index)
    o, f = ours.loc[months].copy(), french.loc[months].copy()
    spread = f"{long_port}-{short_port}"
    o[spread] = o[long_port] - o[short_port]
    f[spread] = f[long_port] - f[short_port]
    rows = {}
    for col in o.columns:
        diff = o[col] - f[col]
        rows[col] = {
            "Ours avg %": o[col].mean() * 100,
            "French avg %": f[col].mean() * 100,
            "Diff avg %": diff.mean() * 100,
            "Corr": o[col].corr(f[col]),
            "Tracking SD %": diff.std() * 100,
        }
    table = pd.DataFrame(rows).T
    table.attrs["months"] = (str(months.min()), str(months.max()), len(months))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a signal's portfolios with Ken French's.")
    parser.add_argument("--signal", required=True, help="file name in src/signals/, e.g. momentum")
    name = parser.parse_args().signal
    spec = load_signal(name)
    if not getattr(spec, "FRENCH_BENCHMARK", None):
        sys.exit(f"src/signals/{name}.py has no FRENCH_BENCHMARK: nothing to compare with.")

    print(f"Ken French file: {spec.FRENCH_BENCHMARK}")
    french = read_french_portfolios(get_french(spec.FRENCH_BENCHMARK))

    print(f"Loading the panel and computing '{name}'...")
    panel = build_panel(spec)
    res = sort_portfolios(panel, name, n_portfolios=10, rebalance=spec.REBALANCE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for w in ("ew", "vw"):
        table = compare(res[w].loc[START:], french[w], spec.LONG, spec.SHORT)
        first, last, n = table.attrs["months"]
        print(f"\n===== {w.upper()}: ours vs French, {first} to {last} ({n} months) =====")
        print(table.to_string())
        table.to_csv(OUT_DIR / f"{name}_{w}.csv")
        spread = f"{spec.LONG}-{spec.SHORT}"
        threshold = [MIN_CORR_SPREAD if row == spread else MIN_CORR for row in table.index]
        weak = table.index[table["Corr"] < threshold].tolist()
        if weak:
            all_ok = False
            print(f"!! Correlation below {MIN_CORR} (spread: {MIN_CORR_SPREAD}) for: {weak}")

    if all_ok:
        print(f"\nAll correlations are above {MIN_CORR} (spread: {MIN_CORR_SPREAD}): "
              "the code reproduces French's portfolios.")
    else:
        print("\nSome correlations are low: look at the rows flagged above.")
    print(f"Results saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
