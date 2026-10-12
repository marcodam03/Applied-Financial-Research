"""Read-only access to Interactive Brokers (TWS / IB Gateway) and local key handling.

Nothing in this module can place orders: the API session is opened with readonly=True.
"""
import os

import pandas as pd


def load_keys(path):
    """Read KEY=VALUE pairs from a local file kept out of version control (see ibkr_keys.env.example)."""
    keys = {}
    if not os.path.exists(path):
        print(f"Key file {path} not found: account id and Flex credentials are empty.")
        return keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip().strip('"').strip("'")
    return keys


def read_portfolio(account, ucits_map, port=7497, client_id=21):
    """Net liquidation value and position values (in base currency) mapped to strategy tickers.

    Positions are matched to the strategy by ISIN first, then by exchange ticker.
    Returns (account_value, {strategy_ticker: value_in_base_ccy})."""
    from ib_async import IB, Contract, util
    try:
        util.startLoop()                       # needed inside Jupyter
    except Exception:
        pass

    ib = IB()
    ib.connect("127.0.0.1", port, clientId=client_id, readonly=True, account=account, timeout=15)
    ib.sleep(2)
    try:
        if account not in ib.managedAccounts():
            raise ValueError(f"Account {account} not visible in this TWS session "
                             f"(visible: {ib.managedAccounts()}).")

        values = ib.accountValues(account)
        base = next(v.currency for v in values if v.tag == "NetLiquidation" and v.currency)
        nav = float(next(v.value for v in values if v.tag == "NetLiquidation" and v.currency == base))
        cash = float(next((v.value for v in values if v.tag == "TotalCashValue" and v.currency == base), 0.0))

        fx = {base: 1.0}
        for v in values:
            if v.tag == "ExchangeRate" and v.currency not in ("", "BASE"):
                fx[v.currency] = float(v.value)

        isin_to_key = {d["isin"]: k for k, d in ucits_map.items()}
        ticker_to_key = {d["ticker"].upper(): k for k, d in ucits_map.items()}

        holdings = {k: 0.0 for k in ucits_map}
        rows, unknown = [], []
        for p in ib.portfolio(account):
            if p.position == 0:
                continue
            c = p.contract
            details = ib.reqContractDetails(Contract(conId=c.conId))
            isin = next((t.value for t in (details[0].secIdList or []) if t.tag == "ISIN"), None) if details else None
            key = isin_to_key.get(isin) or ticker_to_key.get(c.symbol.upper())

            value = p.marketValue * fx[c.currency]
            if key is None:
                key = "?" + c.symbol
                unknown.append(f"{c.symbol} ({isin})")
            holdings[key] = holdings.get(key, 0.0) + value
            rows.append({"Strategy": key, "IBKR ticker": c.symbol, "ISIN": isin,
                         "Quantity": p.position, "Price": round(p.marketPrice, 2),
                         f"Value {base}": round(value, 2), "Weight %": round(100 * value / nav, 1)})
    finally:
        ib.disconnect()

    print(f"Net liquidation value: {base} {nav:,.2f}   |   Cash: {base} {cash:,.2f}\n")
    if rows:
        print(pd.DataFrame(rows).sort_values(f"Value {base}", ascending=False).to_string(index=False))
    else:
        print("No open positions (cash only).")
    if unknown:
        print("\nWARNING: positions not in the UCITS map:", ", ".join(unknown))
        print("They will show up as SELL orders. Add them to the map if they belong to the strategy.")
    return nav, holdings
