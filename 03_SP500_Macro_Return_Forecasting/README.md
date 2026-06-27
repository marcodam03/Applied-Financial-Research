# Oil Shocks, the Yield Curve and U.S. Equity Returns

## Research Question
This project examines whether the structure of financial risk has fundamentally altered since the COVID-19 pandemic. Specifically, it analyzes:
- **Volatility Persistence:** Has market volatility become more entrenched?
- **Diversification:** Did the traditional stock–bond relationship decouple?
- **Tail Risk:** Has the probability of extreme market events increased?

## Methodology & Pipeline
- **Data:** Daily dataset (1990–2026) comprising WTI Crude Oil, S&P 500, and U.S. Treasuries (2Y, 10Y).
- **Transformations:** 
    - Log returns for price series.
    - Duration-based bond-return proxies (D=8.5 for 10Y, D=9.0 for 2Y).
    - Yield curve spread calculation (10Y-2Y) and first-difference analysis.
- **Econometric Framework:**
    - **Diagnostics:** Jarque-Bera (normality), ARCH-LM, and Ljung-Box (volatility clustering).
    - **Modeling:** GARCH for volatility, DCC for time-varying correlations, and VAR models for impulse-response analysis.
    - **Robustness:** Rolling out-of-sample comparison and structural break tests (Markov-switching).

## Repository Structure
- `Figures/` & `Tables/`: Auto-generated output directory for formal reporting.
- `Notebook.ipynb`: Full implementation, from data processing to regime-switching models.
- `Data.xlsx`: Pre-cleaned dataset (internal project reference).

## Contact
Marco D'Amico - [marco.dam03@gmail.com](marco.dam03@gmail.com)
