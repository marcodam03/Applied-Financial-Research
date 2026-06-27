import pandas as pd
from metrics.perf import cagr, ann_vol, sharpe, max_drawdown

def perf_table(bt):
    """
    Build a small performance summary for ML, EW and Momentum portfolios.
    """

    return pd.DataFrame({
        "CAGR":[
            cagr(1+bt["ret_ml"]),
            cagr(1+bt["ret_ew"]),
            cagr(1+bt["ret_mom"])
        ],

        "Vol":[
            ann_vol(bt["ret_ml"]),
            ann_vol(bt["ret_ew"]),
            ann_vol(bt["ret_mom"])
        ],

        "Sharpe":[
            sharpe(bt["ret_ml"]),
            sharpe(bt["ret_ew"]),
            sharpe(bt["ret_mom"])
        ],

        "MaxDD":[
            max_drawdown(bt["cum_ret_ml"]),
            max_drawdown(bt["cum_ret_ew"]),
            max_drawdown(bt["cum_ret_mom"])
        ],
    }, index=["ML Top10%","Equal-Weight","Momentum6M Top10%"])
