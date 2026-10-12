"""From model weights to an order list: optional discretionary overlay and rebalancing amounts."""


def apply_overlay(model_weights, tweaks, journal=""):
    """Add manual tweaks (must net to zero) to the model weights and validate the result.

    model_weights: Series of the latest model weights. tweaks: {ticker: delta}.
    Returns {ticker: final_weight} for weights above 0.01%."""
    print(f"MODEL TARGETS (signal date {model_weights.name.date()}):")
    for a, w in model_weights[model_weights > 0].sort_values(ascending=False).items():
        print(f"  {a:<5} {w*100:6.1f}%")

    final = {}
    for a in set(model_weights.index) | set(tweaks):
        w = max(0.0, model_weights.get(a, 0.0) + tweaks.get(a, 0.0))
        if w > 0.0001:
            final[a] = w

    print("\nFINAL TARGETS:")
    for a, w in sorted(final.items(), key=lambda x: x[1], reverse=True):
        t = tweaks.get(a, 0.0)
        note = f"(model {model_weights.get(a, 0.0)*100:.1f}%, tweak {t*100:+.1f}%)" if abs(t) > 0.0001 else "(model)"
        print(f"  {a:<5} {w*100:6.1f}%  {note}")
    if journal:
        print(f"\nJournal: {journal}")

    total = sum(final.values())
    if abs(total - 1.0) > 0.001:
        print(f"\nWARNING: final weights sum to {total*100:.1f}%. Rebalance the tweaks before trading.")
    else:
        print("\nWeights sum to 100%.")
    return final


def rebalance_orders(targets, holdings, account_value, ucits_map, min_trade=50.0, ccy="EUR"):
    """Amounts to buy/sell per UCITS ETF to move from current holdings to targets.
    Trades below min_trade are ignored. Sells are listed first."""
    orders = []
    for a in set(targets) | set(holdings):
        delta = account_value * targets.get(a, 0.0) - holdings.get(a, 0.0)
        if abs(delta) > min_trade:
            orders.append(("BUY" if delta > 0 else "SELL", a, abs(delta)))

    print(f"Account value: {ccy} {account_value:,.2f}\n")
    if not orders:
        print("No orders needed: the portfolio is already at target.")
        return []

    orders.sort(key=lambda x: x[0], reverse=True)
    print(f"{'Side':<5} | {'ISIN':<13} | {'Ticker':<6} | {'Instrument':<30} | {'Amount (' + ccy + ')':>14}")
    print("-" * 82)
    for side, a, amount in orders:
        d = ucits_map.get(a, {"ticker": a, "isin": "n/a", "desc": "n/a"})
        print(f"{side:<5} | {d['isin']:<13} | {d['ticker']:<6} | {d['desc']:<30} | {amount:>14,.2f}")
    return orders
