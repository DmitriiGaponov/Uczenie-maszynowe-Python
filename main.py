from pathlib import Path
import pandas as pd

from src.preprocessor import DataPreprocessor
from src.visualizer import DataVisualizer
from src.model import FraudModel

from config import RAW_PATH, PROCESSED_PATH, FIGURES_DIR, RESULTS_TXT


def main():
    print("=== PROJEKT: Wykrywanie oszustw finansowych (Fraud Detection) ===")

    # Upewnij się, że foldery na wyniki istnieją
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TXT.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Używany dataset: {RAW_PATH}")

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "Nie znaleziono danych!\n"
            "Dodaj pełny plik do: data/raw/fraud.csv\n"
            "albo upewnij się, że sample istnieje: data/example/fraud_sample.csv"
        )

    # Wczytanie danych
    print("\n[1] Wczytywanie danych...")
    df = pd.read_csv(RAW_PATH)
    print("Liczba rekordów:", len(df))
    print("Kolumny:", list(df.columns))

    # Czyszczenie / preprocessing
    print("\n[2] Czyszczenie i preprocessing...")
    pre = DataPreprocessor()
    df_clean = pre.clean(df) if hasattr(pre, "clean") else pre.process(df)

    # jeśli chcesz zapisać przetworzone dane lokalnie:
    try:
        df_clean.to_csv(PROCESSED_PATH, index=False)
        print(f"[OK] Zapisano dane przetworzone do: {PROCESSED_PATH}")
    except Exception as e:
        print(f"[WARN] Nie udało się zapisać processed csv: {e}")

    # Wizualizacje (jeśli masz takie metody)
    print("\n[3] Wizualizacja wyników / rozkładów...")
    viz = DataVisualizer(output_dir=str(FIGURES_DIR)) if "output_dir" in DataVisualizer.__init__.__code__.co_varnames else DataVisualizer()

    # przykładowo: jeśli masz metody w visualizerze
    if hasattr(viz, "plot_class_distribution"):
        viz.plot_class_distribution(df_clean)
    if hasattr(viz, "plot_correlation"):
        viz.plot_correlation(df_clean)

    #  Modelowanie
    print("\n[4] Trenowanie i ewaluacja modeli...")
    model = FraudModel()

    # W zależności od tego jak masz zrobiony FraudModel – zostawiam elastycznie:
    # - train_evaluate(df) lub fit/predict/score
    results = None
    if hasattr(model, "train_evaluate"):
        results = model.train_evaluate(df_clean)
    elif hasattr(model, "run"):
        results = model.run(df_clean)
    else:
        # Minimalny fallback jeśli masz np. fit + evaluate:
        if hasattr(model, "fit"):
            model.fit(df_clean)
        if hasattr(model, "evaluate"):
            results = model.evaluate(df_clean)

    # Zapis wyników tekstowych (jeśli results to np. dict/str)
    print("\n[5] Zapis wyników...")
    if results is None:
        text = "Brak wyników do zapisania (model nie zwrócił results)."
    elif isinstance(results, str):
        text = results
    else:
        # np. dict -> ładny zapis
        import json
        text = json.dumps(results, ensure_ascii=False, indent=2)

    RESULTS_TXT.write_text(text, encoding="utf-8")
    print(f"[OK] Wyniki zapisane do: {RESULTS_TXT}")

    print("\n=== GOTOWE ===")


if __name__ == "__main__":
    main()
