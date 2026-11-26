"""
Book-to-market signal (Fama and French 1992), as in Ken French's BE/ME portfolios:
book equity of the fiscal year ending in calendar year t-1, divided by the
company's market equity at the end of December t-1. Sorted at the end of June t.

Book equity `be` is built in clean.py (Fama-French definition, missing if <= 0).
Market equity is `me_firm`: all share classes of the company (PERMCO), as French does.
"""
import pandas as pd

COMPUSTAT_COLS = ["be"]
CRSP_COLS = ["me_firm"]
REBALANCE = "annual"
LONG, SHORT = 10, 1  # value (high B/M) beats growth (low B/M)
FRENCH_BENCHMARK = "Portfolios_Formed_on_BE-ME_CSV.zip"


def book_to_market(panel: pd.DataFrame) -> pd.Series:
    # December market equity of year t-1 goes to formation year t (July t - June t+1)
    dec = panel.loc[panel["MthCalDt"].dt.month == 12, ["PERMNO", "MthCalDt", "me_firm"]]
    dec = dec.assign(formation_year=dec["MthCalDt"].dt.year + 1).drop(columns="MthCalDt")
    me_dec = panel[["PERMNO", "formation_year"]].merge(
        dec, on=["PERMNO", "formation_year"], how="left"
    )["me_firm"].to_numpy()  # a left merge keeps the row order of the panel

    bm = panel["be"] / me_dec
    return bm.where(me_dec > 0)
