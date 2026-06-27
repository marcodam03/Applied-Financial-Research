import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from . import config, models

# --- IMPOSTAZIONI GRAFICHE ---
plt.style.use('default') # O 'seaborn-v0_8-whitegrid' se preferisci
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2

def print_pretty_comparison(p_ns, sse_ns, p_nss, sse_nss):
    """Tabella 4: Confronto NS vs NSS (Single Day)"""
    print("\n" + "="*95)
    print(f"| {'NS Parameter':<20} | {'Est. Value':>12} || {'NSS Parameter':<20} | {'Est. Value':>12} |")
    print("-" * 95)
    print(f"| {'Lambda (λ)':<20} | {p_ns[3]:>12.4f} || {'Lambda1 (λ1)':<20} | {p_nss[4]:>12.4f} |")
    print(f"| {'':<20} | {'':>12} || {'Lambda2 (λ2)':<20} | {p_nss[5]:>12.4f} |")
    print(f"| {'Beta 0 (β0)':<20} | {p_ns[0]:>12.6f} || {'Beta 0 (β0)':<20} | {p_nss[0]:>12.6f} |")
    print(f"| {'Beta 1 (β1)':<20} | {p_ns[1]:>12.6f} || {'Beta 1 (β1)':<20} | {p_nss[1]:>12.6f} |")
    print(f"| {'Beta 2 (β2)':<20} | {p_ns[2]:>12.6f} || {'Beta 2 (β2)':<20} | {p_nss[2]:>12.6f} |")
    print(f"| {'':<20} | {'':>12} || {'Beta 3 (β3)':<20} | {p_nss[3]:>12.6f} |")
    print(f"| {'Min Mispricing':<20} | {sse_ns:>12.4f} || {'Min Mispricing':<20} | {sse_nss:>12.4f} |")
    print("="*95 + "\n")

def plot_single_day_spot(params, model_type, date_str):
    """Fig 4 e 5: Spot Rates vs Maturity per il giorno singolo"""
    maturities = np.linspace(0.5, 30, 100)
    if model_type == 'NS':
        rates = models.nelson_siegel_spot(maturities, *params) * 100
        title = f'Spot Rates vs. Maturity (Nelson-Siegel) - {date_str}'
        filename = "Fig4_Spot_NS_Single.png"
    else:
        rates = models.nelson_siegel_svensson_spot(maturities, *params) * 100
        title = f'Spot Rates vs. Maturity (NSS) - {date_str}'
        filename = "Fig5_Spot_NSS_Single.png"

    plt.figure()
    plt.plot(maturities, rates, marker='', linestyle='-', color='tab:blue')
    plt.title(title)
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Spot Rate (%)')
    plt.savefig(os.path.join(config.OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"(Saved: {filename})")

def save_summary_tables(df_results):
    """Tabella 5 e 6: Summary Statistics (Mean, SD, ...)"""
    # Tabella 5: NS
    cols_ns = ['NS_Lam', 'NS_b0', 'NS_b1', 'NS_b2', 'NS_SSE']
    names_ns = ['Lambda', 'Beta0', 'Beta1', 'Beta2', 'Misprice']
    stats_ns = df_results[cols_ns].agg(['mean', 'std', 'median', 'max', 'min']).transpose()
    stats_ns.index = names_ns
    stats_ns.columns = ['Mean', 'SD', 'Median', 'Max', 'Min']
    stats_ns.to_csv(os.path.join(config.OUTPUT_DIR, "Table5_Summary_NS.csv"))

    # Tabella 6: NSS
    cols_nss = ['NSS_Lam1', 'NSS_Lam2', 'NSS_b0', 'NSS_b1', 'NSS_b2', 'NSS_b3', 'NSS_SSE']
    names_nss = ['Lambda1', 'Lambda2', 'Beta0', 'Beta1', 'Beta2', 'Beta3', 'Misprice']
    stats_nss = df_results[cols_nss].agg(['mean', 'std', 'median', 'max', 'min']).transpose()
    stats_nss.index = names_nss
    stats_nss.columns = ['Mean', 'SD', 'Median', 'Max', 'Min']
    stats_nss.to_csv(os.path.join(config.OUTPUT_DIR, "Table6_Summary_NSS.csv"))

    print("\n--- Summary Statistics (Saved to CSV) ---")
    print(stats_nss.to_string(float_format="%.5f"))

def plot_time_series_metrics(df):
    """Fig 8, 9, 10, 11, 12: Parametri e Mispricing nel tempo"""
    dates = pd.to_datetime(df['Date'])

    # Fig 8: Optimal Lambda over time in NS
    plt.figure()
    plt.plot(dates, df['NS_Lam'], marker='o', linestyle='-', color='#1f77b4', markersize=4)
    plt.title('Optimal Lambda over time in NS')
    plt.ylabel('Lambda')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "Fig8_NS_Lambda_Time.png"), dpi=300)
    plt.close()

    # Fig 9: Optimal Lambda1 over time in NSS
    plt.figure()
    plt.plot(dates, df['NSS_Lam1'], marker='o', linestyle='-', color='#1f77b4', markersize=4)
    plt.title('Optimal Lambda1 over time in NSS')
    plt.ylabel('Lambda1')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "Fig9_NSS_Lambda1_Time.png"), dpi=300)
    plt.close()

    # Fig 10: Optimal Lambda2 over time in NSS
    plt.figure()
    plt.plot(dates, df['NSS_Lam2'], marker='o', linestyle='-', color='#1f77b4', markersize=4)
    plt.title('Optimal Lambda2 over time in NSS')
    plt.ylabel('Lambda2')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "Fig10_NSS_Lambda2_Time.png"), dpi=300)
    plt.close()

    # Fig 11: Minimum Misprice over time in NS
    plt.figure()
    plt.plot(dates, df['NS_SSE'], marker='o', linestyle='-', color='gray', markersize=4)
    plt.title('Minimum Misprice over time in NS')
    plt.ylabel('Misprice (SSE)')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "Fig11_NS_Misprice_Time.png"), dpi=300)
    plt.close()

    # Fig 12: Minimum Misprice over time in NSS
    plt.figure()
    plt.plot(dates, df['NSS_SSE'], marker='o', linestyle='-', color='gray', markersize=4)
    plt.title('Minimum Misprice over time in NSS')
    plt.ylabel('Misprice (SSE)')
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "Fig12_NSS_Misprice_Time.png"), dpi=300)
    plt.close()
    print("(Saved: Time Series Plots Figs 8-12)")

def plot_specific_curves_ns(df_results):
    """Fig 13 e 14: Curve NS per giorni specifici della tesi"""
    maturities = np.linspace(0.5, 30, 100)

    # --- Fig 13: 22 Aprile (Low) vs 15 Maggio (High) ---
    target_dates = ['2025-04-22', '2025-05-15']
    colors = ['blue', 'red']
    styles = ['--', '--']

    plt.figure()
    for date_str, col, sty in zip(target_dates, colors, styles):
        row = df_results[df_results['Date'] == pd.to_datetime(date_str)]
        if not row.empty:
            params = [row['NS_b0'].values[0], row['NS_b1'].values[0], row['NS_b2'].values[0], row['NS_Lam'].values[0]]
            rates = models.nelson_siegel_spot(maturities, *params) * 100
            label_date = pd.to_datetime(date_str).strftime('%d-%b-%Y')
            plt.plot(maturities, rates, label=label_date, color=col, linestyle=sty)

    plt.title('Yield Curve in NS (22/04 and 15/05)')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Spot Rate (%)')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(config.OUTPUT_DIR, "Fig13_NS_Curves_Comparison.png"), dpi=300)
    plt.close()

    # --- Fig 14: 28 Apr, 29 Apr, 15 Maggio ---
    target_dates_2 = ['2025-04-28', '2025-04-29', '2025-05-15']
    colors_2 = ['blue', 'gold', 'red'] # Blu, Giallo, Rosso come richiesto
    styles_2 = ['--', '--', '--']

    plt.figure()
    for date_str, col, sty in zip(target_dates_2, colors_2, styles_2):
        row = df_results[df_results['Date'] == pd.to_datetime(date_str)]
        if not row.empty:
            params = [row['NS_b0'].values[0], row['NS_b1'].values[0], row['NS_b2'].values[0], row['NS_Lam'].values[0]]
            rates = models.nelson_siegel_spot(maturities, *params) * 100
            label_date = pd.to_datetime(date_str).strftime('%d-%b-%Y')
            plt.plot(maturities, rates, label=label_date, color=col, linestyle=sty)

    plt.title('Yield Curve in NS (28/04-29/04 and 15/05)')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Spot Rate (%)')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(config.OUTPUT_DIR, "Fig14_NS_Stability_Check.png"), dpi=300)
    plt.close()
    print("(Saved: Fig 13 & 14 Comparison Curves NS)")

def plot_specific_curves_nss(df_results):
    """Grafico NSS Selected Days (Blu, Giallo, Rosso)"""
    # Date tipiche: 29 Apr, 30 Apr, 15 Maggio
    target_dates = ['2025-04-29', '2025-04-30', '2025-05-15']
    colors = ['blue', 'gold', 'red']
    styles = ['--', '--', '--']

    maturities = np.linspace(0.5, 30, 100)
    plt.figure()

    for date_str, col, sty in zip(target_dates, colors, styles):
        row = df_results[df_results['Date'] == pd.to_datetime(date_str)]
        if not row.empty:
            # NSS params: b0, b1, b2, b3, lam1, lam2
            params = [
                row['NSS_b0'].values[0], row['NSS_b1'].values[0], row['NSS_b2'].values[0],
                row['NSS_b3'].values[0], row['NSS_Lam1'].values[0], row['NSS_Lam2'].values[0]
            ]
            rates = models.nelson_siegel_svensson_spot(maturities, *params) * 100
            label_date = pd.to_datetime(date_str).strftime('%d-%b-%Y')
            plt.plot(maturities, rates, label=label_date, color=col, linestyle=sty, linewidth=2.5)

    plt.title('NSS Yield Curves for Selected Days')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Spot Rate (%)')
    # Legenda fuori dal grafico in basso
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True, edgecolor='black')
    plt.tight_layout()

    plt.savefig(os.path.join(config.OUTPUT_DIR, "Fig_NSS_SelectedDays.png"), dpi=300)
    plt.close()
    print("(Saved: NSS Selected Days Plot)")

def plot_comparison_ns_nss_single(params_ns, params_nss, date_str):
    """Confronto diretto NS vs NSS per il giorno singolo (Curve Sovrapposte)"""
    maturities = np.linspace(0.5, 30, 100)

    # Calcolo tassi
    rates_ns = models.nelson_siegel_spot(maturities, *params_ns) * 100
    rates_nss = models.nelson_siegel_svensson_spot(maturities, *params_nss) * 100

    plt.figure()
    plt.plot(maturities, rates_ns, label='Nelson-Siegel', color='blue', linestyle='--', linewidth=2)
    plt.plot(maturities, rates_nss, label='Nelson-Siegel-Svensson', color='red', linestyle='-', linewidth=2)

    plt.title(f'Yield Curve Comparison: NS vs NSS ({date_str})')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Spot Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = "Fig_Comparison_NS_NSS_Single.png"
    plt.savefig(os.path.join(config.OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"(Saved: {filename})")
