import pandas as pd
from . import config

def load_single_day():
    """Carica i dati per l'analisi del giorno singolo"""
    df = pd.read_excel(config.DATA_PATH, sheet_name=config.SHEET_SINGOLO, header=0)
    df.columns = ['id', 'isin', 'coupon', 'maturity', 'clean_price']

    # Conversione tipi
    df['maturity'] = pd.to_datetime(df['maturity'], dayfirst=True)
    df['clean_price'] = pd.to_numeric(df['clean_price'])
    df['coupon'] = pd.to_numeric(df['coupon'])
    return df

def load_30_days_raw():
    """Carica il foglio 'largo' dei 30 giorni"""
    df_raw = pd.read_excel(config.DATA_PATH, sheet_name=config.SHEET_30G, header=0)

    # Rinomina colonne fisse
    fixed_cols = ['id', 'isin', 'coupon', 'maturity']
    df_raw.columns.values[:4] = fixed_cols

    # Preprocessing colonne base
    df_raw['maturity'] = pd.to_datetime(df_raw['maturity'], dayfirst=True)
    df_raw['coupon'] = pd.to_numeric(df_raw['coupon'])

    return df_raw
