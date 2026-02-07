from src.preprocessor import DataPreprocessor
from src.visualizer import DataVisualizer
from src.model import FraudModel

import pandas as pd
from pathlib import Path


RAW_PATH = Path("data/raw/fraud.csv")
PROCESSED_PATH = Path("data/processed/dataset_clean.csv")


def main():

    print("=== PROJEKT: WYKRYWANIE OSZUSTW FINANSOWYCH ===")

    # Wczytanie danych
    print("\n[1] Wczytywanie danych...")

    df = pd.read_csv(RAW_PATH)

    print("Liczba rekordów:", len(df))
    print("Kolumny:", list(df.columns))


    # Przetwarzanie danych
    print("\n[2] Czyszczenie danych...")

    preprocessor = DataPreprocessor()
    df_clean = preprocessor.prepare(df)

    df_clean.to_csv(PROCESSED_PATH, index=False)

    print("Zapisano:", PROCESSED_PATH)


    # Analiza danych (EDA)
    print("\n[3] Analiza danych (EDA)...")

    visualizer = DataVisualizer()
    visualizer.plot_all(df_clean)


    # Trenowanie modeli
    print("\n[4] Uczenie maszynowe...")

    model = FraudModel()
    model.train_and_evaluate(df_clean)


    print("\n=== KONIEC PROGRAMU ===")


if __name__ == "__main__":
    main()
