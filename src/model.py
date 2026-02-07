import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class FraudModel:
    """
    Klasa odpowiedzialna za trenowanie modeli
    i ich ocenę.
    """

    def __init__(self):

        self.out_dir = Path("reports/figures")
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.report_path = Path("reports/results.txt")


    def train_and_evaluate(self, df):

        # Podział na cechy i etykietę
        X = df.drop("isFraud", axis=1)
        y = df["isFraud"]


        # Train / Test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )


        models = {
            "LogReg": LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ),

            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42
            )
        }


        with open(self.report_path, "w", encoding="utf-8") as f:

            f.write("WYNIKI MODELI\n\n")


            for name, model in models.items():

                print("[INFO] Trening:", name)

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)


                acc = accuracy_score(y_test, y_pred)

                f.write(f"Model: {name}\n")
                f.write(f"Accuracy: {acc:.4f}\n\n")

                f.write(
                    classification_report(
                        y_test,
                        y_pred
                    )
                )

                f.write("\n-------\n\n")


                # Macierz pomyłek
                self.plot_confusion_matrix(
                    y_test,
                    y_pred,
                    name
                )


                # Krzywa Precision-Recall
                if hasattr(model, "predict_proba"):

                    y_scores = model.predict_proba(X_test)[:,1]

                    self.plot_pr_curve(
                        y_test,
                        y_scores,
                        name
                    )


        print("Zapisano raport:", self.report_path)


    # Confusion Matrix
    def plot_confusion_matrix(self, y_true, y_pred, name):

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6,5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(f"Macierz pomyłek - {name}")
        plt.xlabel("Klasa przewidziana")
        plt.ylabel("Klasa rzeczywista")

        path = self.out_dir / f"cm_{name}.png"

        plt.savefig(path)
        plt.close()

        print("Zapisano:", path)


    # Precision Recall Curve
    def plot_pr_curve(self, y_true, y_scores, name):

        precision, recall, _ = precision_recall_curve(
            y_true,
            y_scores
        )

        plt.figure(figsize=(7,5))

        plt.plot(recall, precision)

        plt.title(f"Krzywa Precision-Recall - {name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        path = self.out_dir / f"pr_{name}.png"

        plt.savefig(path)
        plt.close()

        print("Zapisano:", path)
