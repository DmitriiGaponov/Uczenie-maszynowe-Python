def test_data_not_empty():
    import pandas as pd
    df = pd.read_csv("data/processed/dataset_clean.csv")
    assert len(df) > 0
