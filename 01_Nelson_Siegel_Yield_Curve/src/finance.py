import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from . import config

def calculate_accrued_and_market_price(df, t0):
    accrued_list = []
    for _, row in df.iterrows():
        maturity = row['maturity']
        coupon_rate = row['coupon']

        # Generazione date a ritroso
        payment_dates = []
        curr = maturity
        limit_date = pd.to_datetime("2000-01-01")
        while curr > limit_date:
            payment_dates.append(curr)
            curr = curr - relativedelta(months=6)
        payment_dates = sorted(payment_dates)
        payment_dates = pd.to_datetime(payment_dates)

        past_dates = payment_dates[payment_dates < t0]
        future_dates = payment_dates[payment_dates >= t0]

        if len(past_dates) == 0 or len(future_dates) == 0:
            accrued_list.append(0.0)
            continue

        last_payment = past_dates[-1]
        next_payment = future_dates[0]

        days_num = (t0 - last_payment).days
        days_den = (next_payment - last_payment).days
        accrued = coupon_rate * days_num / days_den
        accrued_list.append(accrued)

    df['accrued'] = accrued_list
    df['market_price'] = df['clean_price'] + df['accrued']
    return df

def build_cashflows_table(df, t0):
    all_dates = []
    all_bonds = []
    all_cfs = []

    for _, row in df.iterrows():
        bond_id = row['id']
        coupon = row['coupon']
        maturity = row['maturity']

        # Date future
        p_dates = []
        d = maturity
        while d > t0:
            p_dates.append(d)
            d = d - relativedelta(months=6)
        p_dates = sorted(p_dates)

        cfs = [coupon] * len(p_dates)
        if cfs:
            cfs[-1] += config.FACE_VALUE

        all_dates.extend(p_dates)
        all_bonds.extend([bond_id] * len(p_dates))
        all_cfs.extend(cfs)

    return pd.DataFrame({
        'bond_id': all_bonds,
        'payment_date': all_dates,
        'cashflow': all_cfs
    })

def prepare_optimization_vectors(bond_ids, cf_table, t0):
    """Vettorizzazione per velocità"""
    optimized_data = []
    grouped = cf_table.groupby('bond_id')
    t0_np = t0.to_numpy()

    for bid in bond_ids:
        if bid not in grouped.groups:
            optimized_data.append((np.array([]), np.array([])))
            continue
        group = grouped.get_group(bid)
        cfs = group['cashflow'].values
        p_dates = group['payment_date'].values

        taus = (p_dates - t0_np).astype('timedelta64[D]').astype(int) / 365.0
        mask = taus > 0
        optimized_data.append((cfs[mask], taus[mask]))

    return optimized_data
