import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    """
    Klasa odpowiedzialna za przygotowanie danych
    do uczenia maszynowego.
    """

    def __init__(self):

        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()


    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        # Kodowanie typu transakcji
        df["type_enc"] = self.encoder.fit_transform(df["type"])


        # Skalowanie kwoty
        df["amount_scaled"] = self.scaler.fit_transform(
            df[["amount"]]
        )


        # Wybór kolumn do modelu
        selected_cols = [
            "step",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "type_enc",
            "amount_scaled",
            "isFraud"
        ]

        df = df[selected_cols]


        return df
