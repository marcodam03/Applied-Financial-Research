"""Tests of src/data.py with known answers.

Runs offline: the downloads are replaced by fake data, so it checks the logic
(parsing, CHF conversion, bond formula, hedging, splicing, cleaning, saving),
not the websites. Run from the project folder:

    python tests/test_data.py
"""
import os
import sys
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.data as D
from src.config import load_params

CORE_COLUMNS = ["EQ_WORLD", "BANKS_EU", "EQ_CH", "BOND_CHF", "BOND_GLB_H", "REIT_CH", "GOLD", "EMPLOYER"]


# --- 1. Formulas with known answers -------------------------------------------

def test_formulas():
    # CHF conversion: +10% in USD and the USD gains 5% against the CHF -> 1.10 x 1.05 - 1 = 15.5%
    idx = pd.to_datetime(["2020-01-31", "2020-02-29"])
    fx = pd.DataFrame({"USD": [1.00, 1.05]}, index=idx)
    r = D.to_chf(pd.Series([np.nan, 0.10], index=idx), "USD", fx)
    assert abs(r.iloc[1] - 0.155) < 1e-12

    # Bond: yield from 2% to 3%, duration 8 -> 0.02/12 - 8 x 0.01 = -0.0783333
    y = pd.Series([2.0, 3.0], index=idx)
    assert abs(D.synthetic_bond_returns(y, 8).iloc[1] - (0.02 / 12 - 0.08)) < 1e-12

    # Splice: first source has priority, the second fills before AND after it
    i = pd.date_range("2000-01-31", periods=5, freq="ME")
    a = pd.Series([np.nan, 1, 1, np.nan, np.nan], index=i)
    b = pd.Series([2, 2, 2, 2, np.nan], index=i)
    combined, used = D.splice([a, b])
    assert combined.tolist()[:4] == [2, 1, 1, 2] and np.isnan(combined.iloc[4])
    assert used.tolist()[:4] == [1, 0, 0, 1]

    # Only finished months: on 18 Nov 2026 the last month is October; on 1 Dec 2026 it is November
    assert D.last_complete_month("2026-11-18") == pd.Timestamp("2026-10-31")
    assert D.last_complete_month("2026-12-01") == pd.Timestamp("2026-11-30")

    # Cleaning: an impossible +133% month of one ticker becomes NaN, the others are kept
    t = pd.DataFrame({"A": [0.01, 1.33], "B": [0.02, 0.03]}, index=idx)
    cleaned, removed = D.clean_ticker_returns(t, 0.6)
    assert np.isnan(cleaned.loc[idx[1], "A"]) and cleaned.loc[idx[1], "B"] == 0.03 and len(removed) == 1
    print("formulas OK")


# --- 2. File parsers ------------------------------------------------------------

def test_parsers():
    kf = ("Some text\n\n  Average Value Weighted Returns -- Monthly\n,Agric ,Banks \n"
          "192607,   2.37,  -99.99\n192608,   2.23,   1.50\n\n  Annual\n,Agric ,Banks \n1927, 1, 2\n")
    t = D._parse_kf_csv(kf)
    assert list(t.columns) == ["Agric", "Banks"] and len(t) == 2
    assert np.isnan(t["Banks"].iloc[0]) and abs(t["Banks"].iloc[1] - 0.015) < 1e-12
    assert t.index[1] == pd.Timestamp("1926-08-31")

    snb_monthly = '"CubeId";"zimoma"\n\n"Date";"D0";"Value"\n"2000-01";"SARON";"1.5"\n"2000-01";"1TGT";"1.6"\n"2000-02";"SARON";"-0.75"\n'
    s = D._parse_snb_csv(snb_monthly, "SARON")
    assert s.tolist() == [1.5, -0.75]
    snb_daily = '"CubeId";"rendoblid"\n\n"Date";"D0";"Value"\n"2025-07-30";"10J";"0.30"\n"2025-07-31";"10J";"0.35"\n"2025-08-29";"10J";"0.40"\n'
    m = D._to_monthly_last(D._parse_snb_csv(snb_daily, "10J"))
    assert m.tolist() == [0.35, 0.40] and m.index[0] == pd.Timestamp("2025-07-31")
    try:
        D._parse_snb_csv(snb_monthly, "XXX")
        raise AssertionError("a wrong code must raise an error")
    except ValueError as error:
        assert "SARON" in str(error)          # the message lists the codes that exist
    print("parsers OK")


# --- 3. Full pipeline with fake downloads ---------------------------------------

def test_pipeline():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("1999-06-01", "2026-11-18")        # data until "today", in the middle of November
    months = pd.date_range("1990-01-31", "2026-11-30", freq="ME")
    starts = {"URTH": "2012-01-15", "EXV1.DE": "2008-01-15", "EWL": "1999-06-01", "PSPN.SW": "2001-09-01",
              "SPSN.SW": "1999-06-01", "ALLN.SW": "2000-03-01", "GLD": "2004-11-18", "GC=F": "2000-08-30",
              "UBSG.SW": "1999-11-01", "UBS": "2000-05-16", "CSBGC0.SW": "2010-01-01"}

    def fake_yahoo(tickers, start):
        cols = {}
        for tk in tickers:
            p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(days)))), index=days)
            p[p.index < starts.get(tk, "1999-06-01")] = np.nan
            if tk == "PSPN.SW":
                p[p.index >= "2004-05-01"] *= 2.33          # the real Yahoo error: +133% in May 2004
            cols[tk] = p
        return pd.DataFrame(cols).apply(D._to_monthly_last)

    def fake_fred(sid):
        values = {"DEXSZUS": 1.6 - np.linspace(0, 0.75, len(days)), "DEXUSEU": 1.0 + 0.1 * np.sin(np.arange(len(days)) / 300),
                  "DGS10": 6 - np.linspace(0, 4, len(days)), "DTB3": 5 - np.linspace(0, 3, len(days)),
                  }
        return pd.Series(values[sid], index=days, name=sid)

    def fake_kf(file):
        if file == "Developed_3_Factors":
            return pd.DataFrame({"Mkt-RF": rng.normal(0.006, 0.045, len(months)), "RF": 0.002}, index=months)
        kf_months = pd.date_range("1926-07-31", "2026-10-31", freq="ME")
        names = ["Banks", "Insur", "Fin", "Softw", "Hardw", "Chips", "Telcm", "Hlth", "MedEq", "Drugs", "Mach", "ElcEq",
                 "FabPr", "Autos", "Rtail", "Food", "Hshld", "Meals", "Fun", "Oil", "Coal", "Cnstr", "BldMt", "Util"]
        return pd.DataFrame(rng.normal(0.008, 0.06, (len(kf_months), len(names))), index=kf_months, columns=names)

    def fake_snb(cube, code):
        m = pd.date_range("1988-01-31", "2026-10-31", freq="ME")
        if code == "SARON":
            return pd.Series(np.where(m.year.isin(range(2015, 2023)), -0.75, 1.0), index=m)
        if cube == "rendoblim":                              # the real problem: stops in July 2025
            m = m[m <= "2025-07-31"]
            return pd.Series(np.linspace(4, 0.4, len(m)), index=m)
        d = pd.bdate_range("2020-01-01", "2026-10-31")       # daily cube, to today
        return pd.Series(np.linspace(0.5, 0.3, len(d)), index=d)

    real = (D.download_yahoo, D.download_fred, D.download_kenfrench, D.download_snb)
    D.download_yahoo, D.download_fred, D.download_kenfrench, D.download_snb = fake_yahoo, fake_fred, fake_kf, fake_snb
    folder = tempfile.mkdtemp()
    try:
        params = load_params()
        params["data"]["folder"] = folder
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returns, rf = D.update_data(params, verbose=False)

        # names, index, no holes, decimals
        assert list(returns.columns) == CORE_COLUMNS, returns.columns
        assert returns.index.name == "date" and rf.name == "RF_CHF"
        assert returns.notna().all().all() and rf.notna().all()
        assert returns.index.equals(rf.index)
        # the sample ends with the last FINISHED month, never the current one
        assert returns.index.max() == min(pd.Timestamp("2026-10-31"), D.last_complete_month()), returns.index.max()
        assert returns.index.min() == pd.Timestamp("2000-09-30")          # GC=F starts August 2000
        coverage = pd.read_csv(os.path.join(folder, "coverage.csv"))
        bond = coverage[coverage.asset == "BOND_CHF"]
        # BOND_CHF: the ETF wherever it exists (month-end prices), the SNB yield only before it; no monthly averages
        assert set(bond.source) == {"Yahoo CSBGC0.SW", "SNB rendoblim 10J (synthetic, D=9.0)"}, bond.source.tolist()
        assert bond.set_index("source").loc["Yahoo CSBGC0.SW", "to"] == str(returns.index.max().date())
        # the real estate spike is gone: no month near +45%
        assert returns["REIT_CH"].abs().max() < 0.2
        # negative CHF rate: level at end of month t-1 -> months Feb 2015 to Jan 2023
        neg = rf.index[rf < 0]
        assert neg.min() == pd.Timestamp("2015-02-28") and neg.max() == pd.Timestamp("2023-01-31")
        # European banks before 2008 come from the euro-area basket, not from US banks
        banks = coverage[coverage.asset == "BANKS_EU"]
        assert not banks.source.str.contains("Ken French").any()

        # load functions read the saved files and cut the dates
        back = D.load_returns("2010-01-01", "2010-12-31", params=params)
        assert len(back) == 12 and list(back.columns) == CORE_COLUMNS
        assert np.allclose(back.values, returns.loc["2010"].values)
        assert len(D.load_rf("2010-01-01", "2010-12-31", params=params)) == 12

        # sector menu: saved separately, same dates as the core universe, no holes
        sectors = D.load_sector_returns(params=params)
        assert list(sectors.columns) == list(params["data"]["sectors"]) and sectors.index.equals(returns.index)
        assert sectors.notna().all().all()

        # offline: all downloads fail -> the saved raw copies give the same result
        def offline(*args, **kwargs):
            raise ConnectionError("no internet")
        D.download_yahoo = D.download_fred = D.download_kenfrench = D.download_snb = offline
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            returns2, rf2, _ = D.build_data(params, verbose=False)
        assert np.allclose(returns2[CORE_COLUMNS].values, returns.values) and np.allclose(rf2.values, rf.values)
        print("pipeline OK")
    finally:
        D.download_yahoo, D.download_fred, D.download_kenfrench, D.download_snb = real
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    test_formulas()
    test_parsers()
    test_pipeline()
    print("ALL TESTS PASSED")
