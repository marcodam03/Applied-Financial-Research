"""
Intangible capital of each firm, one value per (PERMNO, formation_year), used by
the value and quality scores of the strategy (_qvi_plus.py, _qvi_ext.py).

Accounting expenses most investment in intangibles: R&D and the part of SG&A that
builds brands, customer relationships, organisation and human capital. So book
equity and assets miss this capital, and earnings are lowered by it. Following
Peters and Taylor (2017) and Eisfeldt, Kim and Papanikolaou (2022), it is rebuilt
with the perpetual inventory method, fiscal year by fiscal year:
    knowledge capital     K_t = (1 - 0.15) K_t-1 + xrd_t
    organisation capital  O_t = (1 - 0.20) O_t-1 + 0.30 x (SG&A without R&D)_t
    SG&A without R&D = xsga - xrd when xsga > xrd (Compustat includes R&D in
    xsga), else xsga; missing xrd or xsga counts as 0.
First year of a firm (or after a gap): K = xrd / (0.15 + 0.10), O = 0.30 SG&A /
(0.20 + 0.10), the steady state with 10% growth (Peters and Taylor 2017).
Formation year t uses the fiscal year ending in t-1, like every annual signal.
"""
import numpy as np
import pandas as pd

from src.signals._yearly import december, yearly

DELTA_RD, DELTA_ORG, ORG_SHARE, GROWTH = 0.15, 0.20, 0.30, 0.10
COMPUSTAT_COLS = ["be", "at", "sale", "cogs", "xsga", "xrd", "oibdp"]
CRSP_COLS = ["me_firm"]


def _perpetual(inv: np.ndarray, new: np.ndarray, delta: float) -> np.ndarray:
    cap = np.empty(len(inv))
    for i in range(len(inv)):
        cap[i] = inv[i] / (GROWTH + delta) if new[i] else (1 - delta) * cap[i - 1] + inv[i]
    return cap


def intangible_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Accounting items + rd, org_inv (investment of the year), k_know, k_org and
    me (December t-1 firm market cap, $ millions)."""
    acc = yearly(panel, ["be", "at", "sale", "cogs", "xsga", "xrd"]).sort_index()
    rd = acc["xrd"].fillna(0).clip(lower=0)
    sga = acc["xsga"].fillna(0).clip(lower=0)
    sga_net = sga.where(sga <= rd, sga - rd)
    acc["rd"] = rd
    acc["org_inv"] = ORG_SHARE * sga_net

    firms = acc.index.get_level_values("PERMNO").to_numpy()
    years = acc.index.get_level_values("formation_year").to_numpy()
    new = np.ones(len(acc), dtype=bool)
    new[1:] = (firms[1:] != firms[:-1]) | (years[1:] != years[:-1] + 1)
    acc["k_know"] = _perpetual(rd.to_numpy(), new, DELTA_RD)
    acc["k_org"] = _perpetual(acc["org_inv"].to_numpy(), new, DELTA_ORG)

    me = december(panel, "me_firm") / 1000  # CRSP thousands -> millions, like Compustat
    acc["me"] = me[~me.index.duplicated()].reindex(acc.index)
    return acc
