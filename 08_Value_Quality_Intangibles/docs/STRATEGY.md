# Cheap and cash-profitable within each industry, intangibles counted, at current prices

Final strategy: `src/signals/qvi_timely.py` (score built in `src/signals/_qvi_ext.py` and
`src/signals/_qvi_plus.py`), config `configs/final.yaml`. Every number below can be
reproduced with the commands of section 9.

## 1. The idea in one sentence

Buy the US stocks that are cheap **for their industry** and generate real **cash** from
their operations, once the accounts are corrected for the investment in intangibles (R&D,
brands, organisation) that accounting treats as a cost, valued at **today's** price; short
the opposite.

## 2. The rules

At the end of every month t, with the accounts of the fiscal year that ended in calendar
year t-1 (used from July t, so they are public) and the market value of month t:

1. **Universe:** US common stocks (NYSE, AMEX, NASDAQ) above the NYSE 20th size
   percentile (no microcaps).
2. **Industry:** each stock gets one of the 12 Fama-French industries from its SIC code.
3. **Intangible capital** (`_intangibles.py`, Peters and Taylor 2017), rebuilt year by year
   with the perpetual inventory method:
   - knowledge capital: K(t) = 0.85 K(t-1) + R&D(t)
   - organisation capital: O(t) = 0.80 O(t-1) + 0.30 x (SG&A without R&D)(t)
4. **Cheapness score.** Inside each industry, rank every firm on four ratios, using the
   market value of the current month, and average the ranks (at least three needed):
   - book value / market value, with book = book equity - goodwill + K + O
     (Eisfeldt, Kim and Papanikolaou 2022)
   - earnings / market value
   - cash flow (earnings + depreciation) / market value
   - EBITDA / enterprise value

   Financials (SIC 6000-6999) use only the first two: for them debt is raw material and
   EBITDA has no meaning.
5. **Quality score.** Inside each industry, rank every firm on cash-based operating
   profitability (Ball, Gerakos, Linnainmaa and Nikolaev 2016) corrected for intangibles:
   (sales - cost of goods - SG&A + R&D + 0.30 x SG&A without R&D - increase in receivables
   - increase in inventory + increase in payables) / (total assets + K + O).
6. **Final score:** the average of the two ranks, ranked again inside the industry
   (industries with at least 10 stocks).
7. **Portfolio:** deciles on NYSE breakpoints. Long decile 10 (cheap and profitable),
   short decile 1 (expensive and unprofitable), value-weighted.
8. **Formed every 3 months** (end of March, June, September, December) and held in
   between: the trade-off between a fresh signal and trading costs (forming every 1 or 6
   months gives similar results, section 6).

**Trading costs** (`src/portfolio/costs.py`): every trade pays half the closing bid-ask
spread (CRSP daily quotes; before 1993 an estimate for NYSE/AMEX stocks,
`src/data/cost_inputs.py`) plus a price impact that grows with the trade's share of the
stock's daily volume, for a fund of $100m per side in 2025 dollars.

## 3. Why it should work

- **Industry-relative cheapness removes a bet that does not pay.** A bank always looks
  cheap next to a software firm, so a plain value portfolio is mostly an industry bet.
  Inside an industry, cheapness is real information (Asness, Porter and Stevens 2000).
- **Cash, not accounting profit.** Profit that never arrives as cash is often an
  accounting choice, and investors overpay for it (Sloan 1996; Ball et al. 2016).
- **Each half protects the other.** Cheap stocks alone are often cheap for a reason (value
  traps); profitable stocks alone are often expensive. Requiring both keeps the cheap
  firms that are healthy (Novy-Marx 2013; Asness, Frazzini and Pedersen 2019).
- **Intangibles fix the accounts where they are most wrong.** R&D and brand building are
  investments, but accounting expenses them, so a pharma or software firm looks both
  expensive and unprofitable. Without the correction the strategy would short good firms
  by mistake.
- **Current prices.** With the price of December t-1 (the usual Fama-French convention) a
  stock that has fallen 40% still looks expensive for up to 18 months. The current price
  makes the cheapness score timely (Asness and Frazzini 2013).

## 4. Results 1963-2015 (the research sample)

Long-short spread (decile 10 minus decile 1), % per month, t-stats in brackets; alphas with
Newey-West standard errors (6 lags). Value-weighted, after trading costs.

| | 1963-2015 | 1963-1994 | 1995-2015 |
|---|---|---|---|
| Gross return | 0.50 (3.96) | 0.50 (3.38) | 0.50 (2.22) |
| **Net return** | **0.33 (2.61)** | 0.27 (1.85) | 0.41 (1.84) |
| CAPM net alpha | 0.42 (3.44) | 0.35 (2.45) | 0.52 (2.40) |
| FF5 net alpha | 0.36 (2.55) | -0.06 (-0.50) | 0.55 (2.26) |
| **FF6 net alpha** | **0.53 (3.77)** | 0.02 (0.12) | 0.77 (4.03) |
| **q5 net alpha** | **0.41 (2.58)** | 0.24 (1.33) | 0.62 (2.81) |
| q5 + short-term reversal net alpha | 0.36 (2.3) | | |

Turnover about 9% of each side per month, trading costs 0.17% per month, net Sharpe 0.36.
Equal-weighted: gross 0.64 (5.79) but net only 0.08 (0.69): the small stocks that the
equal-weighted portfolio buys cost too much to trade.

**Comparison with the annual version** (`qvi_intan_full`: same score, December prices,
sorted once a year at the end of June; `python scripts/compare_strategies.py`):

| 1963-2015, VW, net | Final (current prices, every 3 months) | Annual version |
|---|---|---|
| Net return | 0.33 (2.61) | 0.46 (3.64) |
| FF6 net alpha | 0.53 (3.77) | 0.45 (3.13) |
| q5 net alpha | **0.41 (2.58)** | 0.24 (1.60) |

The annual version earns more, but more of its return is explained by the factors; the
final version has the alpha against the hardest benchmark (q5).

## 5. Holdout 2016-2025 (true out-of-sample, run once)

The rules above were frozen before the years 2016-2025 were looked at (`run_signal.py`
hides them unless `--holdout` is given). The holdout was run once, on 2026-10-28
(`holdout_log.txt`; `configs/final.yaml` was called `fast_quarterly.yaml` then).

| VW, net of costs | 1963-2015 | **2016-2025** | 1963-2025 |
|---|---|---|---|
| Net return | 0.33 (2.61) | **0.73 (2.17)** | 0.39 (3.31) |
| CAPM net alpha | 0.42 (3.44) | **0.95 (3.6)** | 0.50 (4.4) |
| FF5 net alpha | 0.36 (2.55) | **0.70 (2.3)** | 0.42 (3.5) |
| FF6 net alpha | 0.53 (3.77) | **0.75 (2.7)** | 0.58 (4.8) |
| q5 net alpha | 0.41 (2.58) | **0.58 (2.4)** | 0.36 (2.5) |
| q5 + STR net alpha | 0.36 (2.3) | **0.61 (2.6)** | 0.32 (2.1) |

Out of sample the strategy did better than in sample. Positive in 8 of 10 calendar years
(net long-short): 2016 +22%, 2017 +2%, 2018 +16% (market -5%), 2019 +21%, 2020 -17%,
2021 +18%, 2022 +10% (market -20%), 2023 +19%, 2024 -6%, 2025 +5%. Trading costs were
0.02% per month (spreads are narrow today). Equal-weighted: net 0.26 (0.80), not
significant.

## 6. Robustness (1963-2015, VW, net)

| Variant | Net % | FF6 net alpha | q5 net alpha |
|---|---|---|---|
| Main (no microcaps, every 3 months) | 0.33 | 0.53 (3.77) | 0.41 (2.58) |
| With microcaps | 0.37 | 0.55 (4.14) | 0.45 (2.99) |
| Large caps only (above the NYSE median) | 0.29 | 0.48 (3.47) | 0.30 (1.99) |
| Formed every month | 0.25 | 0.46 (3.50) | 0.36 (2.34) |
| Formed every 6 months | 0.29 | 0.48 (3.30) | 0.39 (2.36) |
| Every 3 months + turnover buffer | 0.32 | 0.38 (3.41) | 0.25 (1.98) |

**Fund size and cost model** (`run_capacity.py`, q5 net alpha 1963-2015): spread only
0.42 (2.67); square-root price impact 0.40 (2.56); $100m per side 0.41 (2.58); $500m 0.35
(2.22); $1bn 0.28 (1.79); $2.5bn 0.09 (0.55). The alpha survives up to about $500m per
side.

**Luck, or a few years?** (`analyse_strategy.py`)
- Block bootstrap 95% interval of the net mean: 0.09% to 0.59% per month.
- Leave-one-year-out: the t-stat stays between 2.28 and 2.89.
- Without the best 3 years (1970, 2011, 2008) the t-stat falls to 1.6: a good part of the
  return comes from a few crisis years.

**Overfitting** (`overfit_tests.py`):
- Placebo: the score shuffled across stocks each month, 20 times. Gross q5 alpha 0.56
  (t 3.49) for the real score, 0.01 on average for the placebos, none above 0.18.
- Random halves of the stocks (3 splits): the q5 net alpha is positive in all 6 halves and
  the FF6 net alpha has a t between 2.6 and 3.7 in all of them.
- Neighbouring parameters (table above): every variant keeps a positive alpha.

## 7. How it behaves

- Correlation with the market -0.26; FF6 loadings: market -0.15, investment (CMA) +0.40,
  momentum -0.23.
- It pays in bad times: +1.19% per month when the market falls, -0.29% when it rises;
  +2.09% in the 10% worst market months; +31% in 2008, +28% in 2001.
- As a fund (cash in T-bills plus the long-short, 1963-2015): 8.4% per year, volatility
  10.9%, maximum drawdown -24%. Added to the US market (market + 1 x strategy), the Sharpe
  ratio goes from 0.39 to 0.61.

## 8. Limits

- Before 1995 there is no FF6 or q5 alpha: the alpha against the factor models comes from
  1995 onwards (and the holdout).
- A good part of the 1963-2015 return comes from a few crisis years.
- Equal-weighted it does not survive trading costs: it works on large, liquid stocks.
- Short-selling costs (borrow fees, availability) are not in the cost model.

## 9. How to reproduce

```
python src/data/clean.py                  # once, needs the WRDS files in data/raw/
python scripts/run_final_analysis.py      # backtest + every analysis above (about an hour)
python scripts/compare_strategies.py      # final vs annual version
python scripts/run_signal.py --signal qvi_timely --config configs/final.yaml --holdout
                                          # the holdout: already run once, see holdout_log.txt
```
Results go to `results/` (not in git): `results/signals/qvi_timely/`,
`results/analysis/`, `results/capacity/`, `results/overfit/` and
`results/charts/performance.html`.
