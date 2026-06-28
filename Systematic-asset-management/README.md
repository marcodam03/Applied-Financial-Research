# Quantitative Investment Strategies

This repository contains two systematic investment frameworks developed for research and execution purposes.

## Repository Structure

* `systematic-tactical-equity.ipynb`: A systematic approach to equity rotation, utilizing trend-following and correlation-based diversification.
* `multi-asset-absolute-return.ipynb`: A structural alpha model focusing on top-n asset selection and dynamic weight adjustment.

---

## 1. Systematic Tactical Equity Engine
This notebook implements a quantitative engine for equity rotation.

* **Core Logic**:
    * **Trend Filter**: Filters assets based on a 200-day Simple Moving Average (SMA).
    * **Momentum Signal**: Uses 3-month momentum to gauge short-term acceleration.
    * **Tournament Model**: Applies a correlation threshold (0.80) to remove redundant assets and selects top-ranked performers based on momentum.
* **Execution**: Includes logic for dynamic capping and cash management using SHV as a defensive asset.

## 2. Multi-Asset Absolute Return Framework
This notebook focuses on identifying structural alpha through asset selection and volatility management.

* **Core Logic**:
    * **Structural Alpha Signal**: Utilizes a 100-day SMA trend filter and 3-month momentum acceleration logic.
    * **Selection Methodology**: Uses a top-n selection process combined with momentum hysteresis buffers to minimize excessive turnover.
    * **Risk Controls**: Implements an asset cap (50%) and a defensive allocation to cash (SHV) during low-conviction regimes.
* **Outputs**: Generates a performance analytics report and a trade execution blotter for UCITS-compliant instruments.

---

## Technical Requirements

The frameworks are built in Python. Required dependencies:
- `yfinance`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

To set up the environment:
```bash
pip install yfinance pandas numpy matplotlib seaborn