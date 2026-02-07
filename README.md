# Projekt Zaliczeniowy: Wykrywanie Oszustw Finansowych (Fraud Detection)

Autor: Dmitrii Gaponov

---

## 1. Opis Projektu

Celem projektu jest stworzenie aplikacji wykorzystującej uczenie maszynowe (Machine Learning) do automatycznego wykrywania podejrzanych transakcji finansowych na podstawie danych historycznych.

System analizuje informacje o typie transakcji, kwocie, saldzie kont przed i po operacji oraz innych cechach opisujących przelewy. Na tej podstawie modele uczą się rozpoznawać wzorce charakterystyczne dla oszustw finansowych.

Projekt został zaimplementowany w języku Python z wykorzystaniem bibliotek do analizy danych oraz uczenia maszynowego. Zastosowano modularną strukturę projektu, co ułatwia jego rozwój, testowanie oraz utrzymanie.

---

## 2. Realizacja Wymagań

W pierwszym etapie dane są wczytywane z pliku CSV znajdującego się w katalogu `data/raw`. Następnie wykonywana jest wstępna analiza poprawności danych.

Kolejnym krokiem jest przetwarzanie danych, obejmujące:

- usuwanie zbędnych kolumn,
- kodowanie zmiennych kategorycznych,
- skalowanie wartości liczbowych.

Przetworzone dane zapisywane są w katalogu `data/processed`.

Następnie wykonywana jest analiza eksploracyjna danych (EDA), w ramach której generowane są wykresy rozkładu klas oraz macierze korelacji.

Ostatnim etapem jest trenowanie modeli, ich ewaluacja oraz zapis wyników do raportów.

Projekt zawiera również testy jednostkowe w katalogu `tests`.

---

## 3. Analiza Danych (EDA)

Analiza eksploracyjna danych pozwala na lepsze zrozumienie struktury zbioru danych oraz identyfikację potencjalnych problemów.

W ramach EDA analizowany jest rozkład klas zmiennej `isFraud`, który wskazuje na silną nierównowagę danych, gdzie zdecydowana większość transakcji jest poprawna.

Tworzona jest również macierz korelacji, która umożliwia ocenę zależności pomiędzy cechami numerycznymi.

Dodatkowo generowane są wykresy w skali logarytmicznej, które poprawiają czytelność danych przy dużej dysproporcji wartości.

Wszystkie wykresy zapisywane są w katalogu `reports/figures`.

---

## 4. Modelowanie i Wektoryzacja

Dane dzielone są na zbiór treningowy oraz testowy w proporcji 80% do 20%.

Przed trenowaniem modele otrzymują dane po wcześniejszym przetworzeniu.

W projekcie wykorzystano następujące algorytmy:

- Logistic Regression z wagami klas,
- Random Forest Classifier.

Modele trenowane są na zbiorze treningowym, a następnie oceniane na zbiorze testowym.

Dla każdego modelu generowane są:

- raporty klasyfikacji,
- macierze pomyłek,
- krzywe Precision-Recall.

Wyniki zapisywane są w pliku `reports/results.txt`.

---

## 5. Testy

Projekt zawiera testy jednostkowe w katalogu `tests`.

Testy sprawdzają poprawność:

- wczytywania danych,
- przetwarzania danych,
- przygotowania zbiorów,
- inicjalizacji modeli.


## 6. Wyniki i Ewaluaca

Do oceny jakości modeli wykorzystano metryki:

- Accuracy,
- Precision,
- Recall,
- F1-score,
- Confusion Matrix.

Najlepsze wyniki osiągnęły modele Random Forest oraz Logistic Regression z wagami klas.

Modele wykazują wysoką skuteczność w wykrywaniu oszustw, jednak ze względu na nierównowagę danych szczególną uwagę zwrócono na minimalizację liczby fałszywych negatywów.

Uzyskane wyniki potwierdzają skuteczność zastosowanego podejścia w identyfikacji podejrzanych transakcji finansowych.

## 7. Uruchomienie projektu

Instalacja data/raw/fraud.csv: https://www.kaggle.com/datasets/ealaxi/paysim1

Instalacja zależności: pip install -r requirements.txt

Uruchomienie programu: python main.py

