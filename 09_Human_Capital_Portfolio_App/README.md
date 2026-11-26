# Beyond the Portfolio

QARM II group project, HEC Lausanne (autumn 2026).

Investment advice for the savings of a Swiss employee, built around everything else the client owns: the
career (the value today of all the salaries still to come) and the pension fund. A bank gives every balanced
client about 60% growth and 40% bonds. For someone of working age the savings are only a few percent of their
wealth, and a steady salary already pays like a bond. So the app:

1. **Treats the job as a bond.** If 74% of what you own already behaves like a bond, your savings need fewer
   bonds: bonds in the savings = profile's bond share × (1 − bond-like share of total wealth).
2. **Avoids a second bet on your industry.** If part of the pay or employer shares depend on the client's
   industry, the savings buy none of it.
3. **Shows that it worked.** A month-by-month backtest since 2005 for someone like the client, compared with
   the typical advice on total wealth.

## Methods

| What | How | Code |
|---|---|---|
| Career and pension | Human capital as the present value of salaries; bond beta from the salary duration, sector beta = bonus share; pension fund with the legal BVG rules | `src/career.py`, `src/optimizer.py` |
| Covariance | Ledoit-Wolf shrinkage | `src/optimizer.py` |
| Expected returns | Black-Litterman: implied returns of the world market portfolio, optional client view that fades out over 3 years | `src/optimizer.py`, `src/allocation.py` |
| Portfolio | Risk budgeting (cyclical coordinate descent) with budgets by risk profile and Black-Litterman Sharpe ratio; locked employer shares as a fixed holding | `src/allocation.py` |
| Loss limit | Gaussian value at risk: a bad year (1 in 20) must stay within the profile's tolerance plus the savings still to come | `src/allocation.py` |
| Plan by age | The plan solved again every 2 years with the expected balance sheet | `src/allocation.py` |
| Backtest | Walk-forward since September 2005, 60-month estimation window, quarterly rebalancing, trading costs; certainty equivalent on total wealth; circular block bootstrap for p-values | `src/backtest.py`, `src/allocation.py` |

Universe (19 assets, monthly returns in CHF): the 12 Fama-French industries and Swiss equities; Swiss real
estate, gold and commodities; Swiss government bonds, global government bonds and corporate bonds hedged to CHF.
All parameters are in `config/params.yaml`; the code contains no hard-coded numbers.

## How to run the app

Requirements: Git and Python 3 (tested with Python 3.13).

macOS / Linux:

```bash
git clone https://github.com/marcodamico03/QARM2.git
cd QARM2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.precompute
streamlit run app.py
```

Windows (PowerShell):

```powershell
git clone https://github.com/marcodamico03/QARM2.git
cd QARM2
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.precompute
streamlit run app.py
```

`python -m src.precompute` is optional: it computes the backtests of the example clients in advance (about a
minute), so the app opens with everything ready. All the data is saved in the repository, so the app works
without downloading anything.

## Project structure

The code was built in five steps; the comments in the code refer to them.

| Path | Step | What it contains |
|---|---|---|
| `app.py` | 5 | The Streamlit app (one page and four tabs) |
| `src/data.py` | 1 | Downloads, cleaning, conversion to CHF, hedging |
| `src/career.py` | 2 | Career model: human capital, bond and sector betas |
| `src/optimizer.py` | 3 | Ledoit-Wolf, Black-Litterman, pension fund; also version 1 of the advice (mean-variance on total wealth), which the app no longer uses |
| `src/backtest.py` | 4 | Client path month by month, metrics, certainty equivalent, block bootstrap |
| `src/allocation.py` | 3-4 | Version 2, the advice of the app: risk budgeting around the client's job, plan by age, backtest of the strategies |
| `src/dashboard.py`, `src/jobs.py`, `src/precompute.py` | 5 | Results saved on disk and backtests run in the background, so the app never freezes |
| `src/theme.py` | 5 | Look of the app, dark and light |
| `config/params.yaml` | | Every parameter and assumption |
| `data/` | | Saved data (`raw/` as downloaded, CSV files in CHF) |
| `tests/` | | Tests with known answers: `python tests/test_app.py` and one file per module |
