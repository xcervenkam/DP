import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def plot_class_distribution(y: pd.Series, title: str = "Rozdělení tříd") -> None:
    counts = y.value_counts().sort_index()
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Třída")
    plt.ylabel("Počet zápasů")
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame, metric: str) -> None:
    plot_df = results_df.sort_values(metric, ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["model"], plot_df[metric])
    plt.title(f"Porovnání modelů podle metriky: {metric}")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm_df: pd.DataFrame, title: str = "Confusion matrix") -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_df.values, display_labels=cm_df.columns)
    disp.plot()
    plt.title(title)
    plt.tight_layout()
    plt.show()