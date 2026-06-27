import pandas as pd
from src import config, data_loader, finance, models, plotting
import os

def run_single_day():
    print("--- 3. CROSS-SECTIONAL ANALYSIS: SINGLE DAY (May 20) ---")
    df = data_loader.load_single_day()
    df = finance.calculate_accrued_and_market_price(df, config.T0_SINGLE)
    cf_table = finance.build_cashflows_table(df, config.T0_SINGLE)
    bonds_data = finance.prepare_optimization_vectors(df['id'].values, cf_table, config.T0_SINGLE)

    # Calibrazione
    p_ns, sse_ns = models.calibrate_model(df['id'].values, df['market_price'].values, bonds_data, 'NS')
    p_nss, sse_nss = models.calibrate_model(df['id'].values, df['market_price'].values, bonds_data, 'NSS')

    if p_ns is not None and p_nss is not None:
        # Tabella 4
        plotting.print_pretty_comparison(p_ns, sse_ns, p_nss, sse_nss)
        # Fig 4 e 5
        plotting.plot_single_day_spot(p_ns, 'NS', config.T0_SINGLE.strftime('%d-%b-%Y'))
        plotting.plot_single_day_spot(p_nss, 'NSS', config.T0_SINGLE.strftime('%d-%b-%Y'))
        # GRAFICO DI CONFRONTO DIRETTO
        plotting.plot_comparison_ns_nss_single(p_ns, p_nss, config.T0_SINGLE.strftime('%d-%b-%Y'))

def run_30_days():
    print(f"\n--- 4. TIME-SERIES ANALYSIS: 30 DAYS ---")
    df_raw = data_loader.load_30_days_raw()
    price_cols = df_raw.columns[4:]
    results = []

    df_base = df_raw[['id', 'isin', 'coupon', 'maturity']].copy()

    for i, day_col in enumerate(price_cols):
        curr_date = config.START_DATE + pd.Timedelta(days=i)
        print(f"Processing {day_col} ({curr_date.date()})...", end="\r")

        df_day = df_base.copy()
        df_day['clean_price'] = pd.to_numeric(df_raw[day_col], errors='coerce')
        df_day = df_day.dropna(subset=['clean_price'])
        if df_day.empty: continue

        df_day = finance.calculate_accrued_and_market_price(df_day, curr_date)
        cf_table = finance.build_cashflows_table(df_day, curr_date)
        bonds_data = finance.prepare_optimization_vectors(df_day['id'].values, cf_table, curr_date)

        # Calibrazione NS e NSS
        p_ns, sse_ns = models.calibrate_model(df_day['id'].values, df_day['market_price'].values, bonds_data, 'NS')
        p_nss, sse_nss = models.calibrate_model(df_day['id'].values, df_day['market_price'].values, bonds_data, 'NSS')

        # Salvataggio dati grezzi per Tabelle 7 e 8
        row = {'Date': curr_date}

        # Parametri NS: b0, b1, b2, lam
        if p_ns is not None:
            row.update({
                'NS_b0': p_ns[0], 'NS_b1': p_ns[1], 'NS_b2': p_ns[2],
                'NS_Lam': p_ns[3], 'NS_SSE': sse_ns
            })

        # Parametri NSS: b0, b1, b2, b3, lam1, lam2
        if p_nss is not None:
            row.update({
                'NSS_b0': p_nss[0], 'NSS_b1': p_nss[1], 'NSS_b2': p_nss[2], 'NSS_b3': p_nss[3],
                'NSS_Lam1': p_nss[4], 'NSS_Lam2': p_nss[5], 'NSS_SSE': sse_nss
            })

        results.append(row)

    df_res = pd.DataFrame(results)

    # 1. Salva Tabelle 7 e 8 (Serie complete)
    path_full = os.path.join(config.OUTPUT_DIR, "Tables_7_8_Full_Results.csv")
    df_res.to_csv(path_full, index=False)
    print(f"\nCompleted! Full time-series saved to {path_full}")

    # 2. Genera Tabelle 5 e 6 (Statistiche)
    plotting.save_summary_tables(df_res)

    # 3. Genera Grafici Serie Storiche (Fig 8-12)
    plotting.plot_time_series_metrics(df_res)

    # 4. Genera Curve Confronto Giorni Specifici (Fig 13-14 e NSS selected)
    plotting.plot_specific_curves_ns(df_res)
    plotting.plot_specific_curves_nss(df_res)

if __name__ == "__main__":
    run_single_day()
    run_30_days()
