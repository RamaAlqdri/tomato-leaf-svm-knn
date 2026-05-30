"""Compare SVM and KNN cross-validation and test results."""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

try:
    from . import config
    from .utils import ensure_directories, project_path, write_markdown
except ImportError:  # pragma: no cover - direct script execution
    import config  # type: ignore
    from utils import ensure_directories, project_path, write_markdown  # type: ignore


METRIC_COLUMNS = {
    "accuracy": "accuracy_mean",
    "precision": "precision_macro_mean",
    "recall": "recall_macro_mean",
    "f1": "f1_macro_mean",
    "auc": "auc_ovr_macro_mean",
}


def load_cv_results() -> pd.DataFrame:
    svm_path = project_path("reports/svm_results.csv")
    knn_path = project_path("reports/knn_results.csv")
    if not svm_path.exists() or not knn_path.exists():
        raise FileNotFoundError("SVM/KNN result CSV files not found. Run training first.")

    svm_df = pd.read_csv(svm_path)
    svm_df["method"] = "SVM"
    svm_df["setting"] = "RBF"

    knn_df = pd.read_csv(knn_path)
    knn_default_df = knn_df[knn_df["is_default_k"] == True].copy()
    knn_default_df["method"] = "KNN"
    knn_default_df["setting"] = "k=5"

    common_columns = ["method", "setting", "fold", *METRIC_COLUMNS.values()]
    return pd.concat([svm_df[common_columns], knn_default_df[common_columns]], ignore_index=True)


def plot_metric(comparison_df: pd.DataFrame, metric_name: str, output_name: str) -> None:
    column = METRIC_COLUMNS[metric_name]
    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison_df, x="fold", y=column, hue="method")
    plt.ylim(0, 1)
    plt.xlabel("K-Fold Cross Validation")
    plt.ylabel(metric_name.capitalize())
    plt.title(f"Model Comparison - {metric_name.capitalize()}")
    plt.tight_layout()
    plt.savefig(project_path(f"reports/figures/{output_name}"), dpi=200)
    plt.close()


def write_results_summary(comparison_df: pd.DataFrame) -> None:
    display_df = comparison_df.copy()
    for column in METRIC_COLUMNS.values():
        display_df[column] = display_df[column].map(lambda value: f"{value:.4f}")

    best_idx = comparison_df["accuracy_mean"].idxmax()
    best_row = comparison_df.loc[best_idx]

    pivot_rows = []
    for fold in config.CV_FOLDS:
        fold_df = comparison_df[comparison_df["fold"] == fold]
        if fold_df.empty:
            continue
        svm = fold_df[fold_df["method"] == "SVM"].iloc[0]
        knn = fold_df[fold_df["method"] == "KNN"].iloc[0]
        pivot_rows.append(
            {
                "K-Fold": f"{fold}-Fold",
                "KNN Accuracy": knn["accuracy_mean"],
                "KNN Precision": knn["precision_macro_mean"],
                "KNN Recall": knn["recall_macro_mean"],
                "KNN F1": knn["f1_macro_mean"],
                "SVM Accuracy": svm["accuracy_mean"],
                "SVM Precision": svm["precision_macro_mean"],
                "SVM Recall": svm["recall_macro_mean"],
                "SVM F1": svm["f1_macro_mean"],
            }
        )
    pivot_df = pd.DataFrame(pivot_rows)
    pivot_display = pivot_df.copy()
    for column in pivot_display.columns:
        if column != "K-Fold":
            pivot_display[column] = pivot_display[column].map(lambda value: f"{value:.4f}")

    test_eval_path = project_path("reports/test_evaluation_summary.csv")
    test_section = "Belum ada hasil evaluasi test set. Jalankan `python main.py --evaluate`."
    if test_eval_path.exists():
        test_df = pd.read_csv(test_eval_path)
        test_display = test_df.copy()
        for column in test_display.columns:
            if column != "model":
                test_display[column] = test_display[column].map(
                    lambda value: "NA" if pd.isna(value) else f"{float(value):.4f}"
                )
        test_section = test_display.to_markdown(index=False)

    content = f"""# Ringkasan Hasil SVM vs KNN

## Tabel Perbandingan Cross Validation

{pivot_display.to_markdown(index=False)}

## Hasil Evaluasi Test Set

{test_section}

## Model Terbaik Berdasarkan Cross Validation

Model terbaik berdasarkan rata-rata accuracy adalah **{best_row['method']} ({best_row['setting']})**
pada **{int(best_row['fold'])}-Fold Cross Validation** dengan:

- Accuracy: {best_row['accuracy_mean']:.4f}
- Precision macro: {best_row['precision_macro_mean']:.4f}
- Recall macro: {best_row['recall_macro_mean']:.4f}
- F1 macro: {best_row['f1_macro_mean']:.4f}
- AUC OVR macro: {best_row['auc_ovr_macro_mean']:.4f}

## Catatan Interpretasi

Perbandingan utama menggunakan KNN default `k=5`, sesuai rancangan awal. File
`reports/knn_results.csv` tetap menyimpan eksperimen `k=3,5,7,9` untuk analisis tambahan.
Hasil dapat berbeda dari jurnal karena jurnal menggunakan Orange dan tidak menjelaskan detail
ekstraksi fitur citra. Project ini memakai fitur eksplisit RGB histogram, HSV histogram, GLCM, dan LBP.
"""
    write_markdown("reports/results_summary.md", content)


def compare_results() -> pd.DataFrame:
    ensure_directories()
    comparison_df = load_cv_results()
    comparison_df.to_csv(project_path("reports/results_comparison.csv"), index=False)
    plot_metric(comparison_df, "accuracy", "model_comparison_accuracy.png")
    plot_metric(comparison_df, "precision", "model_comparison_precision.png")
    plot_metric(comparison_df, "recall", "model_comparison_recall.png")
    write_results_summary(comparison_df)
    return comparison_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SVM and KNN results.")
    parser.parse_args()
    compare_results()


if __name__ == "__main__":
    main()
