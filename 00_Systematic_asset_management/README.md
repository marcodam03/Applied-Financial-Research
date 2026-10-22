# Systematic Asset Management

Two rules-based ETF strategies, from backtest to a monthly paper-trading process on Interactive Brokers.

| | **Safe** | **Alpha Chaser** |
|---|---|---|
| Objective | Capital preservation, multi-asset | Capture the strongest US equity trends |
| Universe | 10 ETFs: US equity, EM, gold, Treasuries, TIPS, IG credit | 7 equity ETFs (S&P 500, Nasdaq 100, semis, tech, cons. discr., small caps, ARKK) |
| Eligibility | Price > 200-day SMA and 3-month return > 0 | Price > 100-day SMA and 3-month return > 0 |
| Selection | Correlation filter (0.80, 6 months), up to 5 assets | Top 2 by 3-month momentum, +10 pp bonus for current holdings |
| Sizing | **v3 (from Nov 2026):** inverse 60-day volatility, 25% cap, total equity ≤ 60%. *v1: momentum-weighted, 25% cap (40% when the leader is up > 5% in 3 months)* | 50% / 50%; one eligible ETF: 50% + 50% T-bills |
| Risk-off | Residual and full risk-off in 0-1Y T-bills | 100% T-bills when nothing is eligible |
| Rebalancing | Monthly | Monthly |

Signals use month-end data and are applied to the following month. Costs are modelled per ETF (1-8 bps one way) with a 1-10 bps sensitivity band.

## Results

Backtest net of costs, common sample used for the benchmark tests (Safe 09/2007-08/2026, Alpha Chaser 08/2007-08/2026). Sharpe on excess returns over the 1-month T-bill. Benchmarks have no costs.

**Safe**

| | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| **Safe v3 (net)** | **6.98%** | **6.63%** | **0.84** | **-9.9%** |
| Safe v1 (net) | 8.95% | 9.57% | 0.80 | -14.3% |
| 60/40 SPY/IEF | 8.20% | 9.56% | 0.73 | -29.5% |
| 1/N same universe | 7.70% | 9.52% | 0.68 | -23.2% |
| ERC same universe | 6.12% | 7.37% | 0.66 | -19.8% |
| Naive trend 1/N (10m SMA) | 6.14% | 5.59% | 0.85 | -8.2% |
| S&P 500 | 11.09% | 15.58% | 0.67 | -50.8% |

| Alpha test (Newey-West t-stats) | v3 | v1 |
|---|---|---|
| vs 60/40 | +3.2% (t 2.23) | +4.4% (t 2.20) |
| vs risk parity (ERC, same universe) | +2.8% (t 2.30) | +4.0% (t 2.20) |
| vs naive trend 1/N | +0.8% (t 0.96) | +0.9% (t 0.78) |
| Asset-class factors + time-series momentum (v1) | | +1.2% (t 0.72) |

**Safe v1 → v3.** In the forward test v1 held up to 90% equities (small caps and Nasdaq) with a beta of 1.0 to the S&P 500: weighting by momentum concentrates the portfolio in the assets that rose most, which are also the most volatile, and v1 had no limit on total risk. Three fixes were compared with rules fixed before seeing the results ([`research/safe_v2_research.ipynb`](research/safe_v2_research.ipynb)): a 10% volatility target (v2), weight caps only (v2b) and inverse-volatility sizing (v3). Caps and targets only move the strategy along the same risk/return line (Sharpe ≈ 0.8 for all); v3 changes the weighting itself: *trend decides what to hold, risk decides how much*. It had the lowest drawdown in every stress episode tested:

| Max drawdown (daily) | GFC 2008 | Euro crisis 2011 | Covid 2020 | Rates 2022 | Jul 2026 |
|---|---|---|---|---|---|
| Safe v1 | -16.0% | -5.4% | -11.2% | -7.4% | -5.9% |
| Safe v2b (caps) | -13.9% | -6.2% | -11.0% | -7.4% | -5.0% |
| **Safe v3** | **-11.3%** | **-3.5%** | **-9.3%** | **-5.7%** | **-2.9%** |
| S&P 500 (return) | -46.1% | -7.1% | -10.8% | -18.0% | +0.8% |

The cost is about 2 pp of CAGR versus v1; the Sharpe difference is not statistically significant. v3 runs on the same paper account from the November 2026 rebalancing; returns before that date are v1 and are not restated.

**Safe v3 against the literature** ([`research/safe_all_strategies.ipynb`](research/safe_all_strategies.ipynb)). v3 was then tested against eleven alternatives, one per idea from the literature, on the same 10 ETFs, with the same costs and no leverage. The selection rule was fixed before running them: max drawdown within -12%, then the highest Sharpe in the *worse* of the two halves (2008-2016, 2017-2026); an alternative replaces v3 only if it is ahead by at least 0.05.

| Idea | Alternative tested | Source |
|---|---|---|
| Parameters chosen with hindsight | Walk-forward: SMA length and momentum horizon re-chosen every year on past data only | Goyal & Welch (2008) |
| Turning points | Exposure from 12-month and 2-month trends (100% / 50% / 0%) | Goulding, Harvey & Mazzoleni (2023) |
| Faster exits | Daily exit below the 200-day SMA, without waiting for month end | "Breaking Bad Trends" |
| Strategic bonds | At least 20% in Treasuries and TIPS at all times | practitioner feedback |
| Risk-based weights | Equal risk contribution with sample covariance, and with DCC-GARCH covariance re-estimated yearly | Maillard et al. (2010); Engle (2002) |
| Stock-bond hedge | Up to 15% of idle cash into Treasuries when the stock-bond correlation is below -0.2 | stock-bond correlation literature |
| Credit risk filter | Equity <= 30% when the Baa-Aaa spread is high and rising (FRED, publication lag applied) | Welch & Goyal predictors |
| Volatility targeting | Risky sleeve scaled to 7.5% volatility, no leverage | Moreira & Muir (2017) |
| Multi-horizon trend | Exposure from 1, 3, 6, 12-month trends, inverse-volatility weights, 7.5% target | Hurst, Ooi & Pedersen (2017) |
| Alternative architecture | Risk parity on all 9 ETFs x multi-horizon trend, 7.5% target, 20% bond floor | |

Result (08/2008-08/2026, net of costs): **v3 is selected**: Sharpe 0.88 (SE 0.24), 0.93 and 0.84 in the two halves, max drawdown -7.8%. No alternative beats it by more than noise; the two that differ significantly (daily exit 0.77, bond floor 0.84) are worse. Textbook trend 1/N has the same full-sample Sharpe (0.88) but is less stable (0.77 in its worse half, max drawdown -8.2%). The walk-forward version does not beat the fixed parameters (0.68 vs 0.74 on 2013-2026), so the chosen SMA and momentum windows are not the product of in-sample tuning. Bond-heavy designs (risk parity, bond floor) suffer after 2022, when the stock-bond correlation turned from negative to positive.

*Samples differ slightly between notebooks (start date depends on the longest look-back used), so figures are not identical to the table above.*

**Alpha Chaser**

| | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| Alpha Chaser (net) | 18.06% | 18.35% | 0.92 | -21.5% |
| Nasdaq 100 | 16.21% | 18.97% | 0.82 | -49.7% |
| S&P 500 | 11.11% | 15.55% | 0.67 | -50.8% |
| 1/N same universe | 14.28% | 19.69% | 0.71 | -50.5% |
| **Same engine, broad universe (16 ETFs)** | **12.67%** | 17.54% | **0.69** | -20.5% |

| Alpha test | Original universe | Broad universe |
|---|---|---|
| CAPM alpha vs SPY | +9.9% (t 2.97) | +5.1% (t 1.55) |
| Fama-French 5 + momentum | +10.9% (t 3.32) | +5.7% (t 1.81) |

## What the tests say

- **Safe** beats a 60/40 and a plain risk-parity portfolio on a beta-adjusted basis: the value comes from the trend filter, which exits falling assets that risk parity keeps. It adds no alpha over textbook trend-following: it is an efficient, risk-sized implementation of trend-following, whose value is a much lower drawdown. Volatilities and correlations are re-estimated every month with simple rolling windows (60 days, 6 months); DCC-GARCH forecasts, volatility targeting, macro filters and a bond floor were tested out of sample and did not improve on them.
- **Alpha Chaser** looks very strong on its original universe, including after Fama-French 5 + momentum. Running the identical engine on a universe that includes all S&P 500 sectors cuts CAGR by about a third and makes alpha insignificant at 5%. A large part of the original result comes from choosing, with hindsight, the segments that led the last decade.
- **Safe v3 is robust to the alternatives proposed in the literature**: with only 10 ETFs, long-only and no leverage, the remaining room for improvement is small, and the constraints, not the weighting scheme, are the binding limit.
- Both strategies cut drawdowns sharply versus buy-and-hold, which is the most robust finding.

**Limitations.** Yahoo Finance adjusted prices; US-listed ETFs in the backtest, EUR-listed UCITS equivalents in paper trading (tracking difference and FX not modelled); Alpha Chaser parameters (SMA window, buffer) were chosen in-sample (for Safe, a walk-forward check shows no gain from re-tuning them); ARKK only from 2014; costs are estimates; the deterministic stress scenarios are illustrative.

**Work in progress.** Sector-universe robustness with late-inception ETFs handled explicitly, and a longer paper track record.

## Paper trading (forward test)

Both strategies run on separate IBKR **paper** accounts in EUR with UCITS ETFs. The live notebooks call the same functions as the research notebooks, read positions through a read-only API session, produce the order list (entered manually) and a two-page factsheet from the account's daily NAV ([`factsheets/`](factsheets/)).

As of 12 October 2026 (short history, not statistically meaningful):

| | Since | Return | S&P 500 (EUR) | Max DD |
|---|---|---|---|---|
| Safe (v1) | 2 Apr 2026 | +9.5% | +18.4% | -6.1% |
| Alpha Chaser | 1 May 2026 | +15.3% | +13.2% | -18.6% |

**Suggested reading order for Safe:** `safe_research.ipynb` (v1) → `safe_v2_research.ipynb` (v1 to v3) → `safe_all_strategies.ipynb` (v3 against the literature).

## Repository layout

```
research/   safe_research.ipynb, alpha_research.ipynb   backtests, stress tests, alpha tests
            safe_v2_research.ipynb                      Safe v1 vs v2 / v2b / v3, out-of-sample and stress episodes
            safe_all_strategies.ipynb                   Safe v3 vs 11 literature-based alternatives and 4 benchmarks
live/       safe_live.ipynb, alpha_live.ipynb           monthly rebalancing on the paper accounts
src/        data.py         download and month-end features
            strategies.py   signals and portfolio construction (shared by research and live)
            backtest.py     costs, allocation stats, stress test, sub-periods
            benchmarks.py   60/40, 1/N, ERC, naive trend, TSMOM factor, Newey-West OLS, Sharpe test
            ibkr.py         read-only IBKR session
            execution.py    discretionary overlay and order list
            factsheet.py    NAV history (Flex Web Service) and PDF factsheet
factsheets/ monthly PDF factsheets
```

## Running it

```bash
pip install -r requirements.txt
cd research && jupyter notebook
```

The research notebooks only need internet access (Yahoo Finance, Kenneth French Data Library; `safe_all_strategies.ipynb` also downloads two series from FRED, with a free API key in `ibkr_keys.env` as `FRED_API_KEY=`). The live notebooks also need TWS or IB Gateway with the API enabled, and a local `ibkr_keys.env` (copy `ibkr_keys.env.example`). Downloaded data goes to `data/`, which is not tracked.