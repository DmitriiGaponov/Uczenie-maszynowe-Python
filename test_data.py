import pandas as pd
from pathlib import Path


def test_raw_data_exists():
    """
    Sprawdza czy plik z danymi istnieje
    """
    path = Path("data/raw/fraud.csv")

    assert path.exists(), "Brak pliku fraud.csv w data/raw"


def test_raw_data_not_empty():
    """
    Sprawdza czy dane nie są puste
    """
    df = pd.read_csv("data/raw/fraud.csv")

    assert len(df) > 0, "Plik CSV jest pusty"
