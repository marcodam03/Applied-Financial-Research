"""
Trading costs of single stocks, by size group and period (for the paper / the class).

Run from the project folder (needs data/interim/spreads_clean.parquet from clean.py):
    python scripts/cost_by_size.py

For every stock-month of CRSP (all common stocks, microcaps included):
  half spread      half the closing bid-ask spread at the month end, in bps
                   (= the cost of buying OR selling one dollar of the stock)
  estimated %      share of stock-months whose spread is estimated (NYSE/AMEX before 1993)
  daily volume     median average daily dollar volume, $ millions
  impact $1m       price impact of a $1 million trade with the model of
                   src/portfolio/costs.py: lambda x (participation in %) / 10,
                   lambda = 2.5 micro, 1.5 small, 1.0 large, in bps
  total $1m        half spread + impact of a $1 million trade, in bps
Size groups: NYSE 20th and 50th percentiles of market cap (micro < 20th < small < 50th < large).
Output on screen and in results/paper/cost_by_size.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data.panel import load_spreads  # noqa: E402

LAMBDA = {"micro": 2.5, "small": 1.5, "large": 1.0}
PERIODS = {"1963-1979": ("1963-07", "1979-12"), "1980-1992": ("1980-01", "1992-12"),
           "1993-2000": ("1993-01", "2000-12"), "2001-2015": ("2001-01", "2015-12"),
           "2016-2025": ("2016-01", "2025-12")}


def main() -> None:
    s = load_spreads()
    s = s[s["half_spread"].notna() & s["size_group"].notna()].copy()
    s["size_group"] = s["size_group"].astype(str)
    s["estimated"] = ~s["spread_source"].astype(str).str.contains("quote", case=False, na=False)
    lam = s["size_group"].map(LAMBDA)
    participation = 1e6 / s["adv"].where(s["adv"] > 0) * 100          # % of a day's volume
    s["impact_1m"] = (lam * participation.clip(upper=100) / 10) / 100   # as a return
    s["total_1m"] = s["half_spread"] + s["impact_1m"]
    rows = []
    for label, (a, b) in PERIODS.items():
        p = s[(s["month"] >= pd.Period(a, "M")) & (s["month"] <= pd.Period(b, "M"))]
        for g in ["micro", "small", "large"]:
            x = p[p["size_group"] == g]
            if x.empty:
                continue
            rows.append({"period": label, "size group": g, "stock-months": len(x),
                         "half spread mean (bps)": x["half_spread"].mean() * 1e4,
                         "half spread median (bps)": x["half_spread"].median() * 1e4,
                         "estimated %": x["estimated"].mean() * 100,
                         "daily volume median ($m)": x["adv"].median() / 1e6,
                         "impact $1m median (bps)": x["impact_1m"].median() * 1e4,
                         "total $1m median (bps)": x["total_1m"].median() * 1e4})
    t = pd.DataFrame(rows).set_index(["period", "size group"])
    out = BASE_DIR / "results" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "cost_by_size.csv")
    pd.set_option("display.width", 220)
    print(t.round(1).to_string())
    print("\nSpread sources found:", s["spread_source"].astype(str).value_counts().to_dict())
    print(f"Saved {out / 'cost_by_size.csv'}")


if __name__ == "__main__":
    main()
