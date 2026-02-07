from pathlib import Path

# Preferuj pełny dataset lokalnie. Jeśli nie ma, użyj sample z repo.
RAW_FULL = Path("data/raw/fraud.csv")
RAW_SAMPLE = Path("data/example/fraud_sample.csv")
RAW_PATH = RAW_FULL if RAW_FULL.exists() else RAW_SAMPLE

# Zapis danych przetworzonych (ignorowane przez git)
PROCESSED_PATH = Path("data/processed/dataset_clean.csv")

# Wyniki
FIGURES_DIR = Path("reports/figures")
RESULTS_TXT = Path("reports/results.txt")
