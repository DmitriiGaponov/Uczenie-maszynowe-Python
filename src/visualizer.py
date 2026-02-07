import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class DataVisualizer:
    """
    Klasa do wizualizacji danych.
    """

    def __init__(self):

        self.out_dir = Path("reports/figures")
        self.out_dir.mkdir(parents=True, exist_ok=True)


    def plot_all(self, df):

        self.plot_class_distribution(df)
        self.plot_class_distribution_log(df)
        self.plot_correlation(df)


    # Rozkład klas
    def plot_class_distribution(self, df):

        plt.figure(figsize=(7,5))

        sns.countplot(
            x="isFraud",
            data=df
        )

        plt.title("Rozkład klas (0 = brak oszustwa, 1 = oszustwo)")
        plt.xlabel("Klasa")
        plt.ylabel("Liczba transakcji")

        path = self.out_dir / "rozklad_klas.png"

        plt.savefig(path)
        plt.close()

        print("Zapisano:", path)


    # Rozkład klas (log)
    def plot_class_distribution_log(self, df):

        plt.figure(figsize=(7,5))

        sns.countplot(x="isFraud", data=df)

        plt.yscale("log")

        plt.title("Rozkład klas (skala logarytmiczna)")
        plt.xlabel("Klasa")
        plt.ylabel("Liczba transakcji (log)")

        path = self.out_dir / "rozklad_klas_log.png"

        plt.savefig(path)
        plt.close()

        print("Zapisano:", path)


    # Macierz korelacji
    def plot_correlation(self, df):

        plt.figure(figsize=(10,8))

        corr = df.corr()

        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm"
        )

        plt.title("Macierz korelacji cech")

        path = self.out_dir / "macierz_korelacji.png"

        plt.savefig(path)
        plt.close()

        print("Zapisano:", path)
