import pandas as pd
import os

# Percorsi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "30giorni.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Fogli Excel
SHEET_SINGOLO = "singolo"
SHEET_30G = "aprile"

# Date e Parametri
START_DATE = pd.to_datetime("2025-04-15")
T0_SINGLE = pd.to_datetime("2025-05-20")
FACE_VALUE = 100.0

# Creazione cartella output se non esiste
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
