# Macro regimes in real time and risk parity

Marco D'Amico, Advanced Data Analytics, HEC Lausanne, autumn 2026.

In 2022 stocks and US Treasuries fell together, and the usual hedge of a balanced portfolio stopped working. This project asks two things. First, could an investor have seen the change coming from macro data, using only the numbers that were actually published at the time and not today's revised ones? Second, would a risk-parity portfolio of stocks, bonds, gold and commodities have done better by using that information?

The full answer is in `paper.pdf`, and the presentation is in `recording.mp4`. The short answer is "a little, and not enough to matter".

## What you need

- Python 3.11 or newer, and git.
- A free FRED API key. It takes two minutes: create an account at https://fredaccount.stlouisfed.org, then go to *API Keys* and click *Request API Key*. You get a code of 32 letters and digits.
- Nothing else. Everything runs on the course Nuvolos workspace or on a normal laptop, with no GPU, in about 15 minutes in total.

## Step 1: get the code

Open a terminal (on Nuvolos: *File > New > Terminal*) and run:

```
git clone https://github.com/marcodamico03/Advanced_Data_Analytics.git
cd Advanced_Data_Analytics
pip install -r requirements.txt
```

If you work on your own computer, I suggest a clean environment first, so the pinned versions do not clash with what you already have:

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyterlab
```

On Nuvolos JupyterLab is already installed. pip may print a warning about another package of the workspace (for example gpflow): it does not affect this project.

## Step 2: download the data

Still in the terminal, from the `Advanced_Data_Analytics` folder:

```
python code/01_get_data.py
```

The first time it will ask for your FRED key: paste it and press Enter. The script saves it in a file called `.env`, so it never asks again. `.env` is excluded from git, so your key never ends up on GitHub.

The download takes 3 to 5 minutes and fills `data/raw/`.

(If you prefer, you can also create `.env` yourself before running the script: copy `.env.example` to `.env` and write your key after `FRED_API_KEY=`.)

## Step 3: run the notebooks, in order

Open JupyterLab (on your computer: `jupyter lab`). In the file panel on the left, open the `Advanced_Data_Analytics` folder (on Nuvolos it is inside `/files`; click the refresh button if you do not see it), then the `code/` folder, and open the notebooks one by one. For each one, use *Kernel > Restart Kernel and Run All Cells*, and wait until it finishes before opening the next. Each notebook reads what the previous one saved, so the order matters.

| Notebook | What it does | Time |
|---|---|---|
| `02_clean.ipynb` | builds the macro data as known each month, the returns and the stock-bond correlation | under 1 min |
| `03_regimes.ipynb` | growth/inflation quadrants, PCA, clustering, and the check that regime labels stay consistent | 2-3 min |
| `04_portfolio.ipynb` | risk-parity backtests and Sharpe ratio tests | under 1 min |
| `05_forecast.ipynb` | machine-learning forecasts of next month's correlation, refitted every month | about 5 min |
| `06_figures.ipynb` | every figure and table of the paper | about 20 s |

The notebooks find the project folder by themselves, so it does not matter where JupyterLab was started. If the first cell stops with "cannot find a single project code folder", there are two copies of the project in the same place: delete or move one of them.

## What you get

The figures go to `figures/` and the tables to `tables/` (as `.tex` for the paper and `.csv` to read). They are the exact files used in `paper.pdf`, and they are also on GitHub. So after a run you can check whether your numbers match mine with:

```
git diff --stat tables/
```

Do not worry about small differences, they are expected. All FRED data are frozen at 28 September 2026, but Yahoo prices cannot be frozen. Also, the tree models of `05_forecast` (random forest and boosting) can change in the third decimal from one computer to another, and so can the "No fix" column of `tab_stability`. On Nuvolos, for example, the random forest with market variables gives 4.9% instead of 4.8%. None of this changes a conclusion of the paper. The PDF files in `figures/` always show as changed, because they store their creation date.

## Where the data come from

| Source | What | Licence | In the repository? |
|---|---|---|---|
| ALFRED and FRED (Federal Reserve Bank of St. Louis) | 10 US macro series with all their past releases, the yield spread, the T-bill rate | US government data, public domain | yes, in `data/raw/` |
| FRED | Moody's Baa spread, Chicago Fed financial conditions index | third party, no redistribution | no, the script downloads them |
| Yahoo Finance (yfinance) | daily prices of SPY, TLT, GLD, DBC | Yahoo terms, no redistribution | no, the script downloads them |

`data/raw/sources.csv` lists every series with its link and licence. How the data were cleaned (late vintages, the October 2025 shutdown, weekly jobless claims) is explained in section 4 of the paper and in `02_clean.ipynb`.

## How the repository is organised

```
code/01_get_data.py     download (stage 1)
code/02_clean.ipynb     cleaning (stage 2)
code/03_regimes.ipynb   regimes (stage 3)
code/04_portfolio.ipynb portfolios (stage 4)
code/05_forecast.ipynb  forecasts (stage 5)
code/06_figures.ipynb   figures and tables (stage 6)
code/config.py          shared settings: paths, freeze date, series, random seed
code/helpers.py         functions used by more than one stage
data/raw/               downloaded data (only the public part is on GitHub)
data/processed/         created by the notebooks, not on GitHub
figures/, tables/       created by 06_figures.ipynb
paper.pdf, recording.mp4
```

The AI tools I used are declared in the appendix of the paper.
